import os
import time
import torch
import onnx
import logging

# setting stream logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

def export(model, im, file, opset, simplify):
    """Export model to ONNX with dynamic axes option and simplification optionally"""
    logger.info(f"\nStaring exporting with {onnx._version__()}...")
     f = str(file.with_suffix(".onnx"))
     # to be continued..


