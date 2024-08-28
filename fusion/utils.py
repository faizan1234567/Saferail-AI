import torchprofile
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import time
from thop import profile
from torchvision.models import resnet18

def multi_input_hook(input1, input2):
    return dict(input1=input1, input2=input2)


def calculate_gflops(model, input_size):
    flops, params = get_model_complexity_info(
    model, 
    input_size, 
    input_constructor=multi_input_hook,
    as_strings=True, 
    print_per_layer_stat=True, 
    verbose=True
)

def get_params_count(model, trainable=False):
    if not trainable:
        total_params = sum(
        param.numel() for param in model.parameters()
    )
    else:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params/1e6





def compute_gflops_and_model_size(model):
    # Define two inputs with the correct sizes
    input1 = torch.randn(1, 1, 640, 640)  # Input size
    input2 = torch.randn(1, 1, 640, 640)
    
    # Profile the model with these inputs
    macs, params = profile(model, inputs=(input1, input2), verbose=False)
    
    # Calculate GFLOPS and model size
    GFlops = macs / 1e9  # Convert MACs to GFLOPS (Giga FLoating-point Operations Per Second)
    model_size = params * 4.0 / 1024 / 1024  # Model size in MB (assuming 32-bit float params)
    params_M = params / 1e6  # Parameters in millions (M)
    
    return params_M, model_size, GFlops

@torch.no_grad()
def compute_fps(model, shape, epoch=100, device=None):
    """
    frames per second
    :param shape: input size
    """
    total_time = 0.0

    if device:
        model = model.to(device)
    for i in range(epoch):
        data = torch.randn(shape)
        if device:
            data = data.to(device)

        start = time.time()
        outputs = model(data)
        end = time.time()

        total_time += (end - start)

    return total_time / epoch



def test_model_flops():
    model = resnet18()
    params_M, model_size, gflops = compute_gflops_and_model_size(model)

    print('Number of parameters: {:.2f} M '.format(params_M))
    print('Size of model: {:.2f} MB'.format(model_size))
    print('Computational complexity: {:.2f} GFlops'.format(gflops))

def test_fps():
    model = resnet18()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fps = compute_fps(model, (1, 3, 224, 224), device=device)
    print('device: {} - fps: {:.3f}s'.format(device.type, fps))
