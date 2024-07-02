'''
fuse optical and infrared frames using TARDAL algorithm.
'''
import cv2
import numpy as np
import argparse
import yaml
from typing import Tuple
from pathlib import Path
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys
import os

import torch
from torchvision.transforms import ToTensor, Resize, Grayscale, ToPILImage
import torchvision.transforms as trf
import torchvision.transforms.functional as F
from torch import Tensor

# import loader
from TarDAL.config import ConfigDict, from_dict
from TarDAL.pipeline.fuse import Fuse
from TarDAL.tools.dict_to_device import dict_to_device
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, rgb_to_bgr, ycbcr_to_rgb

# load tensorrt engine file
from run_trt_inference import RunTRT

# Logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt = "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def align_images(img1, img2, hmatrix):
    """
    Align two frames for fusion

    Parameters
    ----------
    img1: Numpy.ndarray
    img2: Numpy.ndarray
    homography: str (Path)

    Return
    ------
    aligned_img2: Numpy.ndarray
    """
    # Apply perspective transformation to img2 (thermal image)
    aligned_img2 = cv2.warpPerspective(img2, hmatrix, (img1.shape[1], img1.shape[0]))
    return aligned_img2

def get_ms(tic, toc):
    return (toc - tic) * 1000

def read_grayscale(frame, transformation=None):    
    """
    read in grayscale format and preprocess the frame

    parameters
    ----------
    frame: Numpy.ndarray
    """
    im = transformation(frame)
    im = F.rgb_to_grayscale(im)
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im

def ycbcr_read(frame, transformation=None) -> Tuple[Tensor, Tensor]:
    """
    read in ycbr format and preporcessin the frame

    parameters
    ----------
    frame: Numpy.ndarray
    """
    img_t = transformation(frame)
    img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
    y, cbcr = torch.split(img_t, [1, 2], dim=0)
    if len(y.shape) ==3:
        y = y[None]
        cbcr = cbcr[None]
    return y, cbcr


def image_fusion(frame1, frame2, homography, fuse, trt, transformation=None):
    """
    Fuse two video frames
    ---------------------

    Parameters
    ----------
    frame1: Numpy.ndarray
    frame2: Numpy.ndarray
    homography: str (path to the homography matrix)
    fuse: Fusion object

    Return
    ------
    fused_frame: Numpy.ndarray
    """
    pre_tic = time.time()
    # since the acquired images are not well aligend, wrap the thermal image on the coordinates of optical image
    frame2_aligned = align_images(frame1, frame2, homography) # 25 ms avg
    # read frame as ycbcr, gray and cbcr channels for post processing
    frame1, cbcr = ycbcr_read(frame1, transformation=transformation)
    frame2_aligned = read_grayscale(frame2_aligned, transformation=transformation) # 40 ms preprocessing block
    assert frame1.shape == frame2_aligned.shape, 'both frames should have the same size'
    pre_toc = time.time()
    pre_processing_time = get_ms(pre_tic, pre_toc)
    

    # run inference on optical and aligned thermal frames
    infer_tic = time.time()
    if not trt:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fused_frame = fuse.inference(ir=frame1.to(device), vi=frame2_aligned.to(device))
    else:
        frame1, frame2_aligned = frame1.permute(0, 1, 2, 3).numpy().astype(np.float16), frame2_aligned.permute(0, 1, 2, 3).numpy().astype(np.float16)
        frames = np.concatenate((frame2_aligned, frame1), axis=1)
        fused_frame = fuse.run_trt_inference(frames)
        # plt.imshow(fused_frame[0], cmap= 'gray')
        # plt.show()
        fused_frame = torch.from_numpy(fused_frame).permute(0, 1, 2, 3)
    infer_toc = time.time()
    infer_time = get_ms(infer_tic, infer_toc)


    post_tic = time.time()
    fused_frame = torch.cat([fused_frame.cpu(), cbcr], dim=1)
    fused_frame = ycbcr_to_rgb(fused_frame)
    fused_frame = fused_frame.numpy().squeeze(0)
    fused_frame = np.transpose(fused_frame, (1, 2, 0))
    fused_frame = cv2.cvtColor(fused_frame, cv2.COLOR_RGB2BGR)
    post_toc = time.time()
    post_time = get_ms(post_tic, post_toc)
    return fused_frame, pre_processing_time, infer_time, post_time

def process_frames(video_path1, video_path2, homography, fuse, write=False, trt = False, transformation=None):
    """
    process the fusion videos

    parameters
    ----------
    video_path1: str
    video_path2: str
    homography: str
    fuse: Fusion object
    """
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    durations = []
    if write:
        frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('videos/output.mp4', codec, fps, (frame_width, frame_height))
    
    while cap1.isOpened() and cap2.isOpened():
        # tic_pre = time.time()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not (ret1 and ret2):
            break
        
        # Perform image fusion
        tic = time.time()
        fused_frame, pre_time, infer_time, post_time = image_fusion(frame1, frame2, homography, fuse, trt, transformation=transformation)
        toc = time.time()
        duration = (toc - tic) * 1000
        durations.append(duration)
        print(f'Total mean: {(np.mean(durations)): .2f} ms', end= "")
        print(f' Pre processing: {pre_time: .2f} ms, Fusion inference: {infer_time: .2f} ms, Post processing: {post_time: .2f} ms')
        # write the video
        if write:
            # np.uint8(image)
            img = cv2.cvtColor(np.uint8(fused_frame), cv2.COLOR_BGR2RGB)
            out.write(img)

        # Display or write the fused frame here
        cv2.imshow('Fused Frame', fused_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap1.release()
    cap2.release()
    if write:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--video1', type = str, default = 'videos/1_optical.mp4', help = 'path to optical video')
    parser.add_argument('--video2', type = str, default = 'videos/1_thermal.mp4', help = 'path to thermal video')
    parser.add_argument('--cfg', default='TarDAL/config/default.yaml', help='config file path')
    parser.add_argument('--homography', type = str, default = 'camera_data/homography.npz', help = 'homography path')
    parser.add_argument('--engine', type = str, default= None, help = 'path to save the generated trt file')
    parser.add_argument('--fp16', action= "store_true",  help = 'use fp16 precisoin')
    parser.add_argument('--batch', type = int, default=32, help = 'batch size')
    parser.add_argument('--write', action = 'store_true', help = 'flag to save the video')
    parser.add_argument('--trt', action = 'store_true', help = 'use tensor rt backend')
    args = parser.parse_args()


    # init config
    logger.info("Initialize model configs")
    config  = args.cfg
    if isinstance(config, str) or isinstance(config, Path):
        config = yaml.safe_load(Path(config).open('r'))
        config = from_dict(config)  # convert dict to object
    else:
        config = config

    # load h matrix
    homography_matrix = np.load(args.homography, allow_pickle = True)
    hmat = homography_matrix['homography']

    # image transforms 
    image_shape = (640, 640, 1)
    frame_transformation = trf.Compose([
                                ToPILImage(),
                                Resize(size = image_shape[:2]),
                                ToTensor()])

   
    # init pipeline
    if not args.trt: 
        # use without tensorrt
        fuse = Fuse(config, mode='inference')
        logger.info("Inference: Without using TensorRT")
        process_frames(args.video1, args.video2, hmat, fuse, args.write, trt=False, transformation=frame_transformation)
    
    else:
        # use tensorrt
        data_type = "fp16" if args.fp16 else "fp32"

        # Image transforms
        transformation = trf.Compose([
                                        Grayscale(num_output_channels = 1),
                                        Resize(size = image_shape[:2]),
                                        ToTensor()])

        # Initialize tensorrt wrapper
        logger.info("Create TensorRT engine instance for inference.")
        trt_wrapper = RunTRT(args.engine, data_type= data_type, batch_size= args.batch, image_shape= image_shape)
        logger.info("Warm up")
        trt_wrapper.warmup(runs=200)
        logger.info("Inference: using TensorRT")
        # Run inference
        process_frames(args.video1, args.video2, hmat, trt_wrapper, args.write, trt=True, transformation=frame_transformation)
