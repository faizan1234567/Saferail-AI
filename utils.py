import os
from pathlib import Path
import pandas as pd
import logging

from typing import Optional
import torch
import onnx

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt = "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def export(model, im, file, dynamic, opset, simplify):
    """Export model to ONNX with dynamic axes option and simplification optionally"""
    logger.info(f"\nStaring exporting with {onnx.__version__}...")
    f = str(file.with_suffix(".onnx"))
    output_names= ["Fused"]
    if dynamic:
        dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
    
    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic or None,
    )

      # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            import onnxsim

            logger.info(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
            onnx.save(model_onnx, f)
        except Exception as e:
            logger.info(f"Simplifier failure: {e}")
    return f, model_onnx