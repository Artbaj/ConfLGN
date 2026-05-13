import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torchlogix.layers import LogicConv2d, LogicDense, OrPooling2d, GroupSum
class LogicNet(nn.Module):
    def __init__(self,
        num_classes=10,
        conv_num=3, 
        dense_num=4, 
        base_dense_dims = [2048, 1280, 640, 320],         
        kernel_multiplier=3, 
        k=16,
        tau=6.5,
        gauss_sigma = 1.5,
        parametrization="raw",
        forward_sampling="gumbel_soft",
        weight_init="random",
        init_method="gaussian",
        parametrization_temperature=1.0,         
        channels =1
    ):
        super().__init__()
        logic_kwargs = {
        "parametrization": parametrization,
        "parametrization_kwargs": {
        "forward_sampling": forward_sampling,
        "weight_init": weight_init,
        "temperature": parametrization_temperature,
       
      },
        "connections": "fixed",
        "connections_kwargs": {"init_method": init_method,"sigma": gauss_sigma },
  }

        if conv_num < 1:
            raise ValueError("conv_num must be at least 1")
        if kernel_multiplier < 1:
            raise ValueError("kernel_multiplier must be at least 1")
        if k < 1:
            raise ValueError("k must be at least 1")
        if dense_num != len(base_dense_dims):
            raise ValueError(
                f"dense_num must be equal to len(base_dense_dims). "
                f"Got dense_num={dense_num}, "
                f"len(base_dense_dims)={len(base_dense_dims)}, "
                f"base_dense_dims={base_dense_dims}."
            )
        def conv_out_dim(size, receptive_field_size, padding, stride=1):
            return (size + 2 * padding - receptive_field_size) // stride + 1

        def pool_out_dim(size, kernel_size=2, stride=2, padding=0):
            return (size + 2 * padding - kernel_size) // stride + 1

        channels = 1
        height = 28
        width = 28
        feature_layers = []
        self.shape_trace = []
        self.sigma = gauss_sigma
        for block_idx in range(conv_num):
            receptive_field_size = 5 if block_idx == 0 else 3
            padding = 0 if block_idx == 0 else 1
            num_kernels = k * (kernel_multiplier ** block_idx)

            feature_layers.append(
                LogicConv2d(
                    in_dim=(height, width),
                    channels=channels,
                    num_kernels=num_kernels,
                    tree_depth=3,
                    receptive_field_size=receptive_field_size,
                    padding=padding,
                     **logic_kwargs,
                )
            )

            conv_height = conv_out_dim(height, receptive_field_size, padding)
            conv_width = conv_out_dim(width, receptive_field_size, padding)
            if conv_height <= 0 or conv_width <= 0:
                raise ValueError(f"conv block {block_idx + 1} produces invalid size {conv_height}x{conv_width}")

            feature_layers.append(OrPooling2d(kernel_size=2, stride=2, padding=0))

            pool_height = pool_out_dim(conv_height)
            pool_width = pool_out_dim(conv_width)
            if pool_height <= 0 or pool_width <= 0:
                raise ValueError(f"pool block {block_idx + 1} produces invalid size {pool_height}x{pool_width}")

            self.shape_trace.append(
                {
                    "block": block_idx + 1,
                    "channels": num_kernels,
                    "conv_size": (conv_height, conv_width),
                    "pool_size": (pool_height, pool_width), 
                }
            )

            channels = num_kernels
            height = pool_height
            width = pool_width

        self.features = nn.Sequential(*feature_layers)
        
        flatten_dim = channels * height * width
        layers = [nn.Flatten()]
        if  (base_dense_dims[-1]*k)% num_classes != 0:
            raise ValueError(
                f"base_dense_dims[-1]*k = {base_dense_dims[-1]*k} must be divisible by "
                f"num_classes = {num_classes} for GroupSum"
                )
        
        for dense_idx in range(dense_num):
            dense_out =k*base_dense_dims[dense_idx]
            if(dense_idx == 0):
                dense_in = flatten_dim
            
            if dense_out * 2 < dense_in:
                raise ValueError(
                    f"out_dim * 2 must be >= in_dim at dense layer {dense_idx + 1}. "
                    f"Got out_dim={dense_out}, "
                    f"in_dim={dense_in}, "
                    f"out_dim*2={dense_out * 2}, "
                    f"flatten_dim={flatten_dim}, "
                    f"shape_trace={self.shape_trace}."
                )
            layers.append(LogicDense(in_dim=dense_in, out_dim=dense_out, **logic_kwargs,))
            dense_in = dense_out        

        layers.append(GroupSum(num_classes, tau=tau))   
        self.classifier = nn.Sequential(        
            *layers
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x