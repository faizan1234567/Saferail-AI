import warnings
warnings.filterwarnings("ignore")

import os
import torch
from typing import Union, Tuple, List
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trf
from torchvision.transforms import Normalize, ToPILImage,Grayscale, Resize, ToTensor
from PIL import Image
import cv2

class cDataset(Dataset):
    """load custom dataset for image fusion."""

    def __init__(self, dir: Path, image_shape: Tuple[int, int] = (640, 640), homography_mat: Union[Path, np.ndarray] = None):
        self.dir = dir
        self.vi = os.path.join(self.dir, 'vi')
        self.ir = os.path.join(self.dir, 'ir')  # change if name is differernt
        self.data_transforms =  trf.Compose([
                                  Grayscale(num_output_channels = 1),
                                  Resize(size = image_shape[:2]),
                                  ToTensor()])

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
        # convert to grayscale and concatenate channelwise
        
        image = torch.cat((ir, vi), dim=0)
        return image


if __name__ == "__main__":
    data_path = "/home/jetson/Faizan/fusion/Saferail-AI/images"
    homography_path = "/home/jetson/Faizan/fusion/Saferail-AI/camera_data/homography.npz"
    dataset = cDataset(data_path, homography_mat=homography_path)
    data_loader = DataLoader(dataset, batch_size=16, shuffle= False)
    batch = next(iter(data_loader))
    print(batch.shape)
    