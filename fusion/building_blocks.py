import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out
    
# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out

# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out
    
