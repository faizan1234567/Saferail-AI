import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from building_blocks import DenseBlock, ConvLayer

# DenseFuse network
class DenseFuse_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(DenseFuse_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        # decoder
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

    def encoder(self, input):
        x1 = self.conv1(input)
        x_DB = self.DB1(x1)
        return x_DB


    def fusion(self, en1, en2, strategy_type='addition'):
        f_0 = (en1[0] + en2[0])/2
        return f_0

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return output
    
    def forward(self, img1, img2):
        """
        calculate features from img1 (vi) and img2 (ir)
        both are 1 channel images
        """

        img1_ftrs = self.encoder(img1)
        img2_ftrs = self.encoder(img2)

        # fuse features using fusion stretegy
        fused_ftrs = self.fusion(img1_ftrs, img2_ftrs, "addition")
        fused_ftrs = fused_ftrs[None, :, :, :]

        # Reconstruct the image
        fused_img = self.decoder(fused_ftrs)
        return fused_img


# test code
if __name__ == "__main__":
    from utils import compute_gflops_and_model_size, get_params_count
    # set input size
    input_size = (1, 1, 640, 640)
    img1 = torch.rand(input_size)
    img2 = torch.rand(input_size)

    # define DenseFuse Net
    model = DenseFuse_net(input_nc=1, output_nc=1)
    import time 
    tic = time.time()
    for _ in range(10):
        fused_img = model(img1, img2)
    toc = time.time()
    print(f'avg duration: {(toc - tic)/10}')
    params_M, model_size, GFlops = compute_gflops_and_model_size(model)
    print(f'Model GFLOPS: {GFlops:.2f}')
    print(f'Model Size: {model_size:.2f}')
    print(f'Model Parameters: {params_M:.2f}')

    # saving a pytorch model
    torch.save(model.state_dict(), "densefuse.pth")
   