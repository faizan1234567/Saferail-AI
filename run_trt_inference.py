""""
Run tensorrt inference using engine file
========================================
python run_tensorrt_inference.py -h

Run tensorrt inference on the images in a directory, or an image.
And, it does not support video inference. 
----------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore")

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
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
    def __init__(self, engine_file: Path, data_type: str = "fp16", batch_size: int = 32,
                 image_shape: Tuple[int, int, int]= (640, 640, 1), img_transforms: torchvision.transforms = None, 
                 homography_mat: Path = None):
        
        self.engine_file = engine_file
        self.data_type = data_type
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.transformations = img_transforms
        self.homograhy_data = np.load(homography_mat)
        self.h_matrix = self.homograhy_data["homography"]

        # sample data load
        dataset = cDataset("images/", transforms= self.transformations,  homography_mat=self.h_matrix)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.target_dtype = np.float16 if self.data_type == "fp16" else np.float32
        self.output = np.empty([self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]], dtype = self.target_dtype)
        
        # set dummy data 
        self.optical_batch, self.infrared_batch, self.processed_optical, self.processed_infrared = self.set_dummy_data()
        
        # allocating device memory
        f = open(self.engine_file, "rb")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # allocate device memory
        self.d_input1 = cuda.mem_alloc(1 * self.optical_batch.nbytes)
        self.d_input2 = cuda.mem_alloc(1 * self.infrared_batch.nbytes)
        self.d_inputs = [self.d_input1, self.d_input2]
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input1), int(self.d_input2), int(self.d_output)]

        self.stream = cuda.Stream()
    
    # create image batch
    def create_img_batch(self):
        # change dtype to float16
        (vi, ir, _) = next(iter(self.data_loader))
        ir_batch, vi_batch = ir.permute(0, 2, 3, 1).numpy().astype(self.target_dtype), vi.permute(0, 2, 3, 1).numpy().astype(self.target_dtype)
        return (ir_batch, vi_batch)
    
    # preprocess the images
    def preprocess_image(self, img):
        norm = Normalize(mean=[0.485], std=[0.229])
        result = norm(torch.from_numpy(img).transpose(0,2).transpose(1,2))
        return np.array(result, dtype=np.float16)

    # create dummy data
    def set_dummy_data(self):
        """
        create dummy infrared and optical image pairs for tensorrt engine file
        """
        infrared_batch, optical_batch= self.create_img_batch()

        assert optical_batch.shape == infrared_batch.shape, "Error: shape mismatch"
        assert optical_batch.dtype == infrared_batch.dtype, "Error: dtype mismatch"

        # preprocess the data
        preprocessed_optical = np.array([self.preprocess_image(image) for image in optical_batch])
        preprocessed_infrared = np.array([self.preprocess_image(image) for image in infrared_batch])

        return (optical_batch, infrared_batch, preprocessed_optical, preprocessed_infrared)
    
    # run inference
    def predict(self, inputs:Tuple[np.ndarray, np.ndarray]):
        # transfer input data to device
        for i in range(len(inputs)):
            cuda.memcpy_htod_async(self.d_inputs[i], inputs[i], self.stream)

        # execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # syncronize threads
        self.stream.synchronize()
        return self.output

    # warmup..
    def warmup(self):
        logger.info("Warming up")
        WARMUP_EPOCHS = 150
        for _ in range(WARMUP_EPOCHS):
            pred = self.predict((self.optical_batch, self.infrared_batch))
        logger.info("Warmup complete!")
    
# fuse the images
def fuse(model, dataset, batch_size = 1, shuffle=False, save_dir = None, target_dtype = "fp16"):
    # define data loader object 
    target_dtype = np.float16 if target_dtype == "fp16" else np.float32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # get the images form the data loader with their unique name
    for vi, ir, img_name in data_loader:
        ir, vi = ir.permute(0, 2, 3, 1).numpy().astype(target_dtype), vi.permute(0, 2, 3, 1).numpy().astype(target_dtype)
        fused = model.predict((vi, ir))
        if len(fused.shape) == 4:
            # Post process the image
            fused = np.squeeze((fused[0]* 255).astype(np.uint8), axis=2)
            if save_dir is not None:
                # Creat fused directory to store images 
                new_save_dir = os.path.join(save_dir, 'fused1')
                os.makedirs(new_save_dir, exist_ok= True)
                img_save_path = os.path.join(new_save_dir, img_name[0])
                io.imsave(img_save_path, fused)




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
    
    
    # visualize inputs for debugging purposes
    trt_runner = RunTRT(engine_file= engine_file, data_type= data_type, batch_size= batch, 
                        image_shape= image_shape, img_transforms= transformation,
                        homography_mat= args.homography)
    
    # dataset for inference and visualizatoin
    dataset = cDataset(args.data, transforms= transformation,  homography_mat=args.homography)
    
    
    if args.data:
        logger.info("Running image fusion on the dataset")
        fuse(trt_runner, dataset, args.batch, save_dir= args.save, target_dtype=data_type)
    else:
        logger.info("now run inference.")
        vi_batch, ir_batch = trt_runner.optical_batch, trt_runner.infrared_batch
        acc_time = 0
        RUNS = 10
        outputs = []
        for _ in range(RUNS):
            tic = time.time()
            output = trt_runner.predict((vi_batch, ir_batch))
            outputs.append(output)
            toc = time.time()
            duration = toc - tic
            acc_time += duration
        total_time = (acc_time/RUNS) * 1000
        logger.info(f'WITH TRT: Avearge time taken to run a batch of {batch} images: {total_time: .3f} ms')

        # visualize input data
        plot = True
        if plot:
            max_len = len(outputs)
            # print(max_len)
            ind = random.randint(0, max_len)
            images = [vi_batch[0], ir_batch[0], outputs[ind][0]]
            titles = ["optical", "thermal", "output"]
            nrows, nclos = 1, 3
            fig, axes = plt.subplots(nrows=nrows, ncols=nclos)

            # Loop through images and titles and plot them
            for k in range(nrows * nclos):
                ax = axes[k]
                ax.imshow(images[k], cmap = 'gray')
                ax.set_title(titles[k])
                ax.axis('off')
                # Adjust layout to prevent overlap
            plt.tight_layout()
            plt.show()
        print('done!!')
