""""
Run tensorrt inference using engine file
========================================
python run_tensorrt_inference.py -h

Run tensorrt inference on the images in a directory, or an image.
And, it does not support video inference. ..
----------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore")

import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
from cuda import cudart
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as trf
from torchvision.transforms import Normalize, ToPILImage,Grayscale, Resize, ToTensor
import torchvision.models as models 
import time
from pathlib import Path
from typing import Tuple, List, Union
import time
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random
from skimage import io

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt = "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type = str, default= None, help = 'path to save the generated trt file')
    parser.add_argument('--data', type = str, default= None, help = 'path to the fusion dataset')
    parser.add_argument('--save', type = str, default= "Tardal/data/m3fd_chunked/", help = 'save fused dataset')
    parser.add_argument('--fp16', action= "store_true",  help = 'use fp16 precisoin')
    parser.add_argument('--batch', type = int, default=32, help = 'batch size')
    parser.add_argument('--homography', type = str, default = 'camera_data/homography.npz', help = 'homography path')
    parser.add_argument('--plot', action= "store_true", help = "plot fusion result")
    opt = parser.parse_args()
    return opt


class cDataset:
    """load custom dataset for image fusion."""

    def __init__(self, dir: Path, transforms: torchvision.transforms, homography_mat: Union[Path, np.ndarray] = None):
        self.dir = dir
        self.vi = os.path.join(self.dir, 'vi')
        self.ir = os.path.join(self.dir, 'ir') #  change if name is differernt
        self.data_transforms = transforms
        if homography_mat is not None:
            if type(homography_mat) is not np.ndarray:
                h_matrix_data = np.load(homography_mat)
                self.h_matrix = h_matrix_data["homography"]
            else:
                self.h_matrix = homography_mat
        else:
            self.h_matrix = None 
        self.vi_images = os.listdir(self.vi)
        self.ir_images = os.listdir(self.ir)

        assert len(self.vi_images) == len(self.ir_images), "Error: infrared and optical should have the same number of images"

    def __len__(self):
        return len(self.vi_images)
    
    def perspective(self, h_mat, img):
        if h_mat is not None:
            img = np.array(img)
            aligned_img = cv2.warpPerspective(img, h_mat, (img.shape[1], img.shape[0]))
            return Image.fromarray(aligned_img)
        else:
            return None


    def __getitem__(self, ind):
        img_name = self.vi_images[ind]
        vi_image = os.path.join(self.vi, self.vi_images[ind])
        ir_image = os.path.join(self.ir, self.ir_images[ind])

        # load images
        vi, ir = Image.open(vi_image), Image.open(ir_image)
        # optionally register the infrared image to the coordinat of optical
        ir_aligned = self.perspective(self.h_matrix, ir)

        # transform images if given
        vi, ir = self.data_transforms(vi), self.data_transforms(ir_aligned)
        return (vi, ir, img_name)



# run tnesorrt engine and customize it tpothe fusion task.
class RunTRT:
    def __init__(self, engine_file: Path, data_type: str = "fp16", batch_size: int = 1,
                 image_shape: Tuple[int, int, int]= (640, 640, 1)):
        
        self.engine_file = engine_file
        self.data_type = data_type
        self.batch_size = batch_size
        self.image_shape = image_shape
         
        
        self.load_engine()
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(self.engineString)          # create inference Engine using Runtime
        if self.engine == None:
            print("Failed building engine!")
        print("Succeeded building engine!") 


        self.nIO = self.engine.num_io_tensors                                                 # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor, all the APIs with "binding" in their name are deprecated
        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.nIO)]               # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and Excution Context are indexed by name, not binding number like TensorRT 8.4 or before
        self.nInput = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(trt.TensorIOMode.INPUT)  # get the count of input tensor
        self.nOutput = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(trt.TensorIOMode.OUTPUT)  # get the count of output tensor

        self.context = self.engine.create_execution_context()                                 # create Excution Context from the engine (analogy to a GPU context, or a CPU process)
        self.context.set_input_shape(self.lTensorName[0], (1, 2, 640, 640))                   # set actual size of input tensor if using Dynamic Shape mode
        
    def load_engine(self):
        self.logger = trt.Logger(trt.Logger.ERROR)                                 # create Logger, avaiable level: VERBOSE, INFO, WARNING, ERRROR, INTERNAL_ERROR
        if os.path.isfile(self.engine_file):                                       # load serialized network and skip building process if .plan file existed
            with open(self.engine_file, "rb") as f:
                self.engineString = f.read()
            if self.engineString == None:
                print("Failed getting serialized engine!")
                return
            print("Succeeded getting serialized engine!")
        else:                                                                       # build a serialized network from scratch
            builder = trt.Builder(logger)                                           # create Builder
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  # create Network
            profile = builder.create_optimization_profile()                         # create Optimization Profile if using Dynamic Shape mode
            config = builder.create_builder_config()                                # create BuidlerConfig to set meta data of the network
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)     # set workspace for the optimization process (default value is total GPU memory)

            inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])   # set inpute tensor for the network
            profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])   # set danamic range of the input tensor
            config.add_optimization_profile(profile)                                # add the Optimization Profile into the BuilderConfig

            identityLayer = network.add_identity(inputTensor)                       # here is only a identity transformation layer in our simple network, which the output is exactly equal to input
            network.mark_output(identityLayer.get_output(0))                        # mark the output tensor of the network

            engineString = builder.build_serialized_network(network, config)        # create a serialized network
            if engineString == None:
                print("Failed building serialized engine!")
                return
            print("Succeeded building serialized engine!")
            with open(self.engine_file, "wb") as f:                                 # write the serialized netwok into a .plan file
                f.write(engineString)
                print("Succeeded saving .plan file!")


    def run_trt_inference(self, inputs: np.ndarray):
        bufferH = []                                                      # prepare the memory buffer on host and device
        bufferH.append(np.ascontiguousarray(inputs))

        for i in range(self.nInput, self.nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(self.lTensorName[i]), dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i]))))

        bufferD = []
        for i in range(self.nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(self.nInput):                                                     # copy input data from host buffer into device buffer
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(self.nIO):
            self.context.set_tensor_address(self.lTensorName[i], int(bufferD[i]))         # set address of all input and output data in device buffer

        self.context.execute_async_v3(0)                                                   # do inference computation

        for i in range(self.nInput, self.nIO):                                              # copy output data from device buffer into host buffer
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        for b in bufferD:                                                                   # free the GPU memory buffer after all work
            cudart.cudaFree(b)
        
        return bufferH[1]

    # Warmup 
    def warmup(self, inputs: Tuple[np.ndarray, np.ndarray] = None, runs: int = 200):
        if inputs == None:
            img = np.random.rand(1, 1, 640, 640).astype(np.float16)
            inputs = np.concatenate((img, img), axis=1)
        logger.info("Warming up")
        for _ in range(runs):
            pred = self.run_trt_inference(inputs)
        logger.info("Warmup complete!")
    

# Plot fusion results
def show_fusion_result(vi, ir, output):
    images = []
    arrays = [vi, ir, output]
    for array in arrays:
        img = np.squeeze(array, axis=(0, 1))
        images.append(img)

    titles = ["optical", "thermal", "output"]
    nrows, nclos = 1, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=nclos)

    for k in range(nrows * nclos):
        ax = axes[k]
        ax.imshow(images[k], cmap = 'gray')
        ax.set_title(titles[k])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Correct format
def create_img_batch(loader, target_dtype = np.float16):
    (vi, ir, img_name) = next(iter(loader))
    ir_batch, vi_batch = ir.permute(0, 1, 2, 3).numpy().astype(target_dtype), vi.permute(0, 1, 2, 3).numpy().astype(target_dtype)
    return (ir_batch, vi_batch, img_name)


if __name__ == "__main__":
    # read args
    args = read_args()


    logger.info("setting fusion pipeline")
    engine_file = args.engine
    data_type = "fp16" if args.fp16 else "fp32"
    batch = args.batch
    image_shape = (640, 640, 1)

    # image transforms
    transformation = trf.Compose([
                                  Grayscale(num_output_channels = 1),
                                  Resize(size = image_shape[:2]),
                                  ToTensor()])
    
    
    # create an instance of TensorRT runtime
    trt_runner = RunTRT(engine_file= engine_file, data_type= data_type, batch_size= batch, 
                        image_shape= image_shape)
    

    # Load the dataset.
    dataset = cDataset(args.data, transforms= transformation,  homography_mat=args.homography)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=False)

    # Get a batch for testing
    ir, vi, image_name = create_img_batch(data_loader, target_dtype= np.float16 if data_type == "fp16" else np.float32)

    # Run warmup before running actual inference.
    WARMUP_RUNS = 100
    trt_runner.warmup((ir, vi), runs= WARMUP_RUNS)

    logger.info("now run inference and average the inference time.")
    acc_time = 0
    RUNS = 10
    for _ in range(RUNS):
        tic = time.time()
        output = trt_runner.run_trt_inference((ir, vi))
        toc = time.time()
        duration = toc - tic
        acc_time += duration
    total_time = (acc_time/RUNS) * 1000
    logger.info(f'WITH TRT: Avearge time taken to run a batch of {batch} images: {total_time: .3f} ms')

    if args.plot:
        show_fusion_result(vi, ir, output)

