"""
Generate an ONNX file from a trained PyTorch model
==================================================

for help 
python generate_onnx.py -h
===================================================
"""

import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Union
from utils import export
import numpy as np
from utils import export
import numpy as np

import torch
import torchvision.models as models 
from torchvision.transforms import Normalize
from torch import Tensor
import logging
import time
import argparse
import yaml
from TarDAL.config import ConfigDict, from_dict
from TarDAL.config import ConfigDict, from_dict
from pathlib import Path
import os
from datetime import datetime
import onnx
import onnx

# from pipeline.fuse import Fuse
from TarDAL.module.fuse.generator import Generator

# meta fusion
from MetaFusion.models.metafusion_net import FusionNet as FusionNetwork

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt = "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def read_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='TarDAL/config/default.yaml', help='config file path')
    parser.add_argument("--weights", type=str, default="TarDAL/weights/v1/tardal-dt.pth", help="model.pt path(s)")
    parser.add_argument('--batch', type = int, default= 1,  help = "batch size")
    parser.add_argument('--model_name', choices=["tardal", "meta_fusion"], type = str, default= "tardal-dt",  help = "Name of the image fusion model")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

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
    return model
    
def load_meta_fusion(model, weights):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        model.load_state_dict(torch.load(weights, map_location=torch.device("cpu")), strict=True)
    else:
        model.load_state_dict(torch.load(weights, map_location=None), strict=True)
    return model



class Pt2ONNX:
    """Export to ONNX from PyTorch"""
    def __init__(self, model, batch_size: int = 1,
                 image_shape: Tuple[int, int, int]= (640, 640, 1), dynamic: bool = False, 
                 opset_version: int = 17) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.image_shape = (self.batch_size,) + image_shape
        self.opset = opset_version
        self.model = model

    def create_dummpy_data(self):
        """
        create dummy dataset for generate an onnx file, the generated datset should be
        related to the fusion. 
        i.e. for image fusion we need a tuple of images (one for optical and one for thermal)
        """
        # create a batch of vi and ir images
        optical = torch.randn(self.image_shape, dtype = torch.float32, requires_grad=True).permute(0, 3, 1, 2)
        infrared = torch.randn(self.image_shape, dtype = torch.float32, requires_grad=True).permute(0, 3, 1, 2)

        assert optical.shape == infrared.shape, "Error: shape mismatch of optical and infrared images"
        assert optical.shape[1] == 1 and infrared.shape[1] ==1, "Error: Should be grayscale images"
        return (infrared, optical)
    
    def torch2onnx(self, model_name):
        """
        convert PyTorch trained model to onnx model
        """
        logger.info("Converting to ONNX from PyTorch \n")
        dir_name = "onnx_files/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        file_name = model_name
        f = Path(os.path.join(dir_name, file_name))
        f = str(f.with_suffix(".onnx"))
        ir, vi = self.create_dummpy_data()
        if model_name == "tardal":
            im = torch.cat((ir, vi), dim=1)
        elif model_name == "meta_fusion":
            if torch.cuda.is_available():
                im = torch.rand(1, 4, 384, 512)
            else:
                im = torch.rand(1, 4, 640, 640)
        try:
            import gc
            gc.collect()
            torch.onnx.export(self.model,
                        im, 
                        f, 
                        verbose= False,
                        export_params=True,
                        do_constant_folding=True, 
                        input_names=["image"],
                        output_names=["fused"])
            
            # Checks
            model_onnx = onnx.load(f)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model   
                     
        except FileNotFoundError:
            logger.info(f"File not Found {f}")
            logger.info(f"File not Found {f}")


if __name__ == "__main__":
    # read command line args
    args = read_args()
    logger.info("Generating ONNX file from a Trained PyTorch model\n")
    if args.model_name == "tardal":
        model = load_tardal(args.weights, args.cfg)
    elif args.model_name == "meta_fusion":
        model = FusionNetwork(block_num=3, feature_out=False)
        model = load_meta_fusion(model, args.weights)
    onnx_converter = Pt2ONNX(model=model, batch_size=args.batch, image_shape=(640, 640, 1), 
    opset_version= 17)
    onnx_converter.torch2onnx(args.model_name)
    logger.info('Conversion done')