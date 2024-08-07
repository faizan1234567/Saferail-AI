import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models, datasets

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import calib
from tqdm import tqdm

print(pytorch_quantization.__version__)
import tensorrt as trt
import numpy as np
import tarfile
import shutil 
import logging
import yaml

from TarDAL.config import ConfigDict, from_dict
from pathlib import Path
import os
from datetime import datetime
import onnx

# from pipeline.fuse import Fuse
from TarDAL.module.fuse.generator import Generator

from run_trt_inference import cDataset
from torch.utils.data import Dataset, DataLoader


# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt = "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def load_tardal_weights(model, ckpt):
    """
    load PyTorch trained weights into the model in the inference mode.
    """
    if 'use_eval' in ckpt:
        ckpt.pop('use_eval')
    model.load_state_dict(ckpt)
    return model 

def load_tardal(weights, cfg):
    """load TarDAL for ONNX conversion"""
    
    # Init config 
    logger.info("Initializing Configuration settings! \n")
    if isinstance(cfg, str) or isinstance(cfg, Path):
        config = yaml.safe_load(Path(cfg).open('r'))
        config = from_dict(config)  # convert dict to object
    else:
        config = cfg
    
    # Init model
    logger.info("Initializing model \n")
    f_dim, f_depth = config.fuse.dim, config.fuse.depth
    model = Generator(dim=f_dim, depth=f_depth)
    
    # Load weights 
    logger.info("Loading model weights to the model \n")
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    ckpt = torch.load(weights, map_location= map_location)
    model = load_tardal_weights(model, ckpt)
    model.eval()
    return model, ckpt


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (ir, vi, image_name) in tqdm(enumerate(data_loader), total=num_batches):
        image = torch.cat((ir, vi), dim=1)
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

if __name__ == "__main__":
    
    
    quant_modules.initialize()
    # data loaders
     # Load the dataset.
    weights = "TarDAL/weights/v1/tardal-dt.pth"
    cfg = "TarDAL/config/default.yaml"
    data_path = "images"
    homography_path = "camera_data/homography.npz"
    image_shape = (640, 640, 1)
    transformation =  transforms.Compose([
                        transforms.Grayscale(num_output_channels = 1),
                        transforms.Resize(size = image_shape[:2]),
                        transforms.ToTensor()])

    dataset = cDataset(data_path, transforms= transformation,  homography_mat=homography_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    
    q_model, ckpt = load_tardal(weights=weights, cfg=cfg)
    q_model = q_model.cuda()
 

    modified_state_dict={}
    for key, val in ckpt.items():
        # Remove 'module.' from the key names
        if key.startswith('module'):
            modified_state_dict[key[7:]] = val
        else:
            modified_state_dict[key] = val

    # Load the pre-trained checkpoint
    q_model.load_state_dict(modified_state_dict)
    
    # Calibrate the model using max calibration technique.
    with torch.no_grad():
        collect_stats(q_model, data_loader, num_batches=16)
        compute_amax(q_model, method="max")
    
    # Save the PTQ model
    torch.save(q_model.state_dict(), "TarDAL/weights/tardal-dt-ptq.pth")
    
    # convert to onnx
    # Set static member of TensorQuantizer to use Pytorch’s own fake quantization functions
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Exporting to ONNX
    dummy_input = torch.randn(1, 2, 640, 640, device='cuda')
    input_names = [ "actual_inputs" ]
    output_names = [ "fused_output" ]
    torch.onnx.export(
        q_model,
        dummy_input,
        "onnx_files/tardal-dt-ptq.onnx",
        verbose=False,
        opset_version=13,
    do_constant_folding = False)
    # now build the file using int8

