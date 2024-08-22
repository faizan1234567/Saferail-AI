#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from datetime import datetime as dt
from glob import glob
import argparse

import calibrate
import cv2
import numpy as np
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from cuda import cudart
from torch.autograd import Variable


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default= "", type = str, help = "path to onnx file")
    parser.add_argument("--engine", default= "", type = str, help = "path to save engine file")
    parser.add_argument("--data", type = str, default="", help= 'paht to the data dir')
    parser.add_argument("--cache", type = str, default="int8.cache", help= './int8.cache')
    parser.add_argument("--fp16", action= "store_true", help= 'use fp16')
    parser.add_argument("--int8", action= "store_true", help="use int8")
    opt = parser.parse_args()
    return opt
    

if __name__ == "__main__":
    # Read cmd args
    args = read_args()
    nCalibration = 1
    cudart.cudaDeviceSynchronize()
    
    # Set randon params
    np.random.seed(31193)
    t.manual_seed(97)
    t.cuda.manual_seed_all(97)
    t.backends.cudnn.deterministic = True


    # Values of some variables
    BATCH=1
    nHeight = 640
    nWidth = 640

    # Parse network, rebuild network and do inference in TensorRT
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrate.Calibrator(args.data, nCalibration, (BATCH, 2, nHeight, nWidth), args.cache)
        
    # Parse ONNX and build engine
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(args.onnx):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(args.onnx, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
    
    # Build engine
    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, [BATCH, 2, nHeight, nWidth], [BATCH, 2, nHeight, nWidth], [BATCH, 2, nHeight, nWidth])
    config.add_optimization_profile(profile)

    # network.unmark_output(network.get_output(0))  # dont remove output
    output_tensor = network.get_output(0)
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(args.engine, "wb") as f:
        f.write(engineString)
        
    # Run using TensorRT engine
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [1, 2, nHeight, nWidth])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    data = np.random.rand(BATCH, 2, 640, 640) # RANDOM INPUT
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nIO):
        print(lTensorName[i])
        print(bufferH[i])

    for b in bufferD:
        cudart.cudaFree(b)

    print("Succeeded running model in TensorRT!")