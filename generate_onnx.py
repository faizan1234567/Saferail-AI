"""
Generate an ONNX file from a trained PyTorch model
==================================================
"""

import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Union
from utils import export
import numpy as np

import torch
import torchvision.models as models 
import torch.onnx 
from torchvision.transforms import Normalize
from torch import Tensor
import logging
import time
import argparse
import yaml
from TarDAL.config import ConfigDict, from_dict
from pathlib import Path
import os
from datetime import datetime

# from pipeline.fuse import Fuse
from TarDAL.module.fuse.generator import Generator

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
    parser.add_argument("--weights", type=str, default="TarDAL/weights/tardal-dt.pth", help="model.pt path(s)")
    parser.add_argument('--batch', type = int, default= 1,  help = "batch size")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX: opset version")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["onnx"],
        help="onnx, engine",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

class Pt2ONNX:
    """Export to ONNX from PyTorch"""
    def __init__(self, trained_weights: Path, cfg: Union[Path, Dict], batch_size: int = 1,
                 image_shape: Tuple[int, int, int]= (640, 640, 1), dynamic: bool = False, 
                 opset_version: int = 17, simplify: bool = False) -> None:
        """
        constructor method
        
        takes trained model weights and configuration file.
        """

        self.trained_weights = trained_weights
        self.cfg  = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.image_shape = (self.batch_size,) + image_shape
        self.dynamic = dynamic
        self.opset = opset_version
        self.simplify = simplify

        # init config
        logger.info("Initializing Configuration settings!")
        if isinstance(self.cfg, str) or isinstance(self.cfg, Path):
            config = yaml.safe_load(Path(self.cfg).open('r'))
            self.config = from_dict(config)  # convert dict to object
        else:
            self.config = self.cfg
        
        # init model
        logger.info("Initializing model")
        f_dim, f_depth = self.config.fuse.dim, self.config.fuse.depth
        self.model = Generator(dim=f_dim, depth=f_depth)

        # load ckpt to the model
        logger.info("Loading model weights to the model")
        map_location = lambda storage, loc: storage
        if torch.cuda.is_available():
            map_location = None
        ckpt = torch.load(trained_weights, map_location= map_location)
        self.load_weights(ckpt)
        self.model.eval()


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
    
    def load_weights(self, weights):
        """
        load PyTorch trained weights into the model in the inference mode.
        """
        if 'use_eval' in weights:
            weights.pop('use_eval')
        self.model.load_state_dict(weights)

    def torch2onnx(self):
        """
        convert PyTorch trained model to onnx model
        """
        logger.info("Converting to ONNX from PyTorch")
        dir_name = "onnx_files/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        file_name = self.trained_weights.split("/")[-1]
        f = Path(os.path.join(dir_name, file_name))
        f = str(f.with_suffix(".onnx"))
        ir, vi = self.create_dummpy_data()
        im = torch.cat((ir, vi), dim=1)
        try:
            f = export(self.model, im, Path(f), dynamic=self.dynamic, opset=self.opset, simplify= self.simplify)
        except FileNotFoundError:
            logger.info(f"File not Found {f}")

if __name__ == "__main__":
    # read command line args
    args = read_args()
    logger.info("Generating ONNX file from a Trained PyTorch model")
    onnx_converter = Pt2ONNX(trained_weights=args.weights, cfg=args.cfg, batch_size=args.batch, image_shape=(640, 640, 1), 
    dynamic= args.dynamic, opset_version= args.opset, simplify= args.simplify)
    onnx_converter.torch2onnx()
    logger.info('Conversion done')