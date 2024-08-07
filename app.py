"""
========================================================
fuse optical and infrared frames using TARDAL algorithm.
========================================================

Author: Muhammad Faizan
All right reserved!

"""
# Basic dependencies 
import cv2
import numpy as np
import argparse
import yaml
from typing import Tuple
from pathlib import Path
import time
import threading
import warnings
warnings.filterwarnings("ignore")
import sys

# Torch dependencies
import torch
from torchvision.transforms import ToTensor, Resize, ToPILImage
import torchvision.transforms as trf
import torchvision.transforms.functional as F
from torch import Tensor

# Image fusion dependencies
from TarDAL.config import from_dict
from TarDAL.pipeline.fuse import Fuse
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, ycbcr_to_rgb

# Object detection 
from run_trt_inference import RunTRT
import imutils
from JetsonYolov5.yoloDet import YoloTRT


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
    # Homography
    aligned_img2 = cv2.warpPerspective(img2, hmatrix, (img1.shape[1], img1.shape[0]))
    return aligned_img2



def get_ms(tic, toc):
    """get time"""
    return (toc - tic) * 1000

def read_grayscale(frame, transformation=None):    
    """
    read in grayscale format and preprocess the frame

    parameters
    ----------
    frame: Numpy.ndarray
    """
    im = transformation(frame).to("cuda")
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
    img_t = transformation(frame).to("cuda")
    img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
    y, cbcr = torch.split(img_t, [1, 2], dim=0)
    if len(y.shape) ==3:
        y = y[None]
        cbcr = cbcr[None]
    return y, cbcr


# Fuse frames
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

    # Align
    frame2_aligned = align_images(frame1, frame2, homography)
    frame1, cbcr = ycbcr_read(frame1, transformation=transformation)
    frame2_aligned = read_grayscale(frame2_aligned, transformation=transformation)
    assert frame1.shape == frame2_aligned.shape, 'both frames should have the same size'

    # Inference
    if not trt:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fused_frame = fuse.inference(ir=frame1.to(device), vi=frame2_aligned.to(device))
        
    else:
        # TensorRT engine
        if frame1.device.type == "cuda" and frame2_aligned.device.type == "cuda":
            frame1, frame2_aligned = frame1.cpu().permute(0, 1, 2, 3).numpy().astype(np.float16), frame2_aligned.cpu().permute(0, 1, 2, 3).numpy().astype(np.float16)
        else:
            frame1, frame2_aligned = frame1.permute(0, 1, 2, 3).numpy().astype(np.float16), frame2_aligned.permute(0, 1, 2, 3).numpy().astype(np.float16)
        frames = np.concatenate((frame2_aligned, frame1), axis=1)
        fused_frame = fuse.run_trt_inference(frames)
        fused_frame = torch.from_numpy(fused_frame).permute(0, 1, 2, 3).to("cuda")
    
    # Post processing
    if fused_frame.device.type == "cuda" and cbcr.device.type == "cuda":
        fused_frame = torch.cat([fused_frame.cpu(), cbcr.cpu()], dim=1)
    else:
        fused_frame = torch.cat([fused_frame, cbcr], dim=1)
    fused_frame = ycbcr_to_rgb(fused_frame)
    fused_frame = fused_frame.numpy().squeeze(0)
    fused_frame = np.transpose(fused_frame, (1, 2, 0))
    fused_frame *= 255
    fused_frame = fused_frame.astype(np.uint8)
    return fused_frame

def process_frames(video_path1, video_path2, homography, fuse, write=False, trt = False, transformation=None, detection_model=None, fusion=True):
    """
    process the fusion videos

    parameters
    ----------
    video_path1: str
    video_path2: str
    homography: str
    fuse: Fusion object
    """
    # FPS display info
    font_scale = 0.5
    thickness = 1
    font_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
 
    
    durations = []
    if write:
        frame_height = 640
        frame_width = 640
        fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('videos/output.mp4',fourcc, fps, (frame_height, frame_width))
    
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not (ret1 and ret2):
            break
        
        tic = time.time()
        # Fuse images if specified
        if fusion:
            fused_frame = image_fusion(frame1, frame2, homography, fuse, trt, transformation=transformation)
        else:
            fused_frame = frame2
        print()    
        if detection_model is not None:
            frame = imutils.resize(fused_frame, width=640)
            detections, t = detection_model.Inference(frame)
        else:
            print("Error in detection model")
            sys.exit(0)
        duration = (time.time() - tic)
        durations.append(duration)
        fps_info = f'Current FPS: {(1/duration): .3f}, Average FPS: {((1/np.mean(durations))): .3f}'
        (text_width, text_height), baseline = cv2.getTextSize(fps_info, font, font_scale, thickness)
        text_pos = (10, text_height + 10)
        cv2.rectangle(frame, (text_pos[0], text_pos[1] - text_height - baseline), (text_pos[0] + text_width, text_pos[1] + baseline), bg_color, cv2.FILLED)
        cv2.putText(frame, fps_info, text_pos, font, font_scale, font_color, thickness, cv2.LINE_AA)
        if write:
            out.write(frame)
        
        cv2.imshow('detection Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap1.release()
    cap2.release()
    if write:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--video1', type = str, default = 'videos/1_optical.mp4', help = 'path to optical video')
    parser.add_argument('--video2', type = str, default = 'videos/1_thermal.mp4', help = 'path to thermal video')
    parser.add_argument('--cfg', default='TarDAL/config/default.yaml', help='config file path')
    parser.add_argument('--cfg', default='TarDAL/config/default.yaml', help='config file path')
    parser.add_argument('--homography', type = str, default = 'camera_data/homography.npz', help = 'homography path')
    parser.add_argument('--engine', type = str, default= None, help = 'path to save the generated trt file')
    parser.add_argument('--write', action = 'store_true', help = 'flag to save the video')
    parser.add_argument('--fusion', action='store_true', help = 'use image fusion')
    args = parser.parse_args()


    # Init config
    logger.info("Initialize model configs")
    config  = args.cfg
    if isinstance(config, str) or isinstance(config, Path):
        config = yaml.safe_load(Path(config).open('r'))
        config = from_dict(config)  # convert dict to object
    else:
        config = config

    # Homography matrix
    homography_matrix = np.load(args.homography, allow_pickle = True)
    hmat = homography_matrix['homography'].astype(np.float32)
    
    # Detector
    yolo_model = YoloTRT(library="JetsonYolov5/yolov5/build/libmyplugins.so", engine="JetsonYolov5/yolov5/build/yolov5s.engine", conf=0.45, yolo_ver="v5")


    # Image transforms
    image_shape = (640, 640, 1)
    frame_transformation = trf.Compose([
                                ToPILImage(),
                                Resize(size = image_shape[:2]),
                                ToTensor()])
    
    use_pt_file = False
    if use_pt_file: 
        fuse = Fuse(config, mode='inference')
        logger.info("Inference: Without using TensorRT")
        process_frames(args.video1, args.video2, hmat, fuse, args.write, trt=False, transformation=frame_transformation, detection_model=yolo_model)
    
    else:
        
        # TensorRT based inference
        data_type = "fp16"
        logger.info("Create TensorRT engine instance for inference.")
        trt_wrapper = RunTRT(args.engine, data_type= data_type, batch_size=1, image_shape= image_shape)
        trt_wrapper.warmup(runs=200)
        logger.info("Inference: using TensorRT")
        # Run inference
        process_frames(args.video1, args.video2, hmat, trt_wrapper, args.write, trt=True, transformation=frame_transformation, detection_model=yolo_model, 
                       fusion=args.fusion)

