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

#TODO: code to be tested
def export(model, im, file, dynamic, opset, simplify):
    """Export model to ONNX with dynamic axes option and simplification optionally"""
    logger.info(f"\nStaring exporting with {onnx._version__()}...")
     f = str(file.with_suffix(".onnx"))
     output_name = ["Fused"]
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

    # Metadata
    d = {"stride": int(max(model.stride)), "names": model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

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





