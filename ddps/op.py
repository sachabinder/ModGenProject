import torch
import torch.nn as nn
import torch.nn.functional as F
from ddps.op_utils import Resizer, Kernel, Blurkernel
from torchvision.transforms.functional import to_pil_image

from functools import partial


class SuperResolutionOperator(nn.Module):
    def __init__(self, in_shape, scale_factor):
        super(SuperResolutionOperator, self).__init__()
        self.scale_factor = scale_factor
        self.down_sample = Resizer(in_shape, 1 / scale_factor)
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)

    def forward(self, x, keep_shape=False, **kwargs):
        x = (x + 1.0) / 2.0
        y = self.down_sample(x)
        y = (y - 0.5) / 0.5
        if keep_shape:
            y = F.interpolate(y, scale_factor=self.scale_factor, mode="bicubic")
        return y

    def transpose(self, y):
        return self.up_sample(y)

    def y_channel(self):
        return 3

    def to_pil(self, y):
        y = (y[0] + 1.0) / 2.0
        y = torch.clip(y, 0, 1)
        y = to_pil_image(y, "RGB")
        return y


class GaussialBlurOperator(nn.Module):
    def __init__(self, kernel_size=61, intensity=3.0):
        super(GaussialBlurOperator, self).__init__()

        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="gaussian", kernel_size=kernel_size, std=intensity
        )
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, data, **kwargs):
        return self.conv(data)

    def y_channel(self):
        return 3

    def transpose(self, data, **kwargs):
        return data

    def to_pil(self, y):
        y = (y[0] + 1.0) / 2.0
        y = torch.clip(y, 0, 1)
        y = to_pil_image(y, "RGB")
        return y


class MotionBlurOperator(nn.Module):
    def __init__(self, kernel_size=61, intensity=0.5):
        super(MotionBlurOperator, self).__init__()
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="motion", kernel_size=kernel_size, std=intensity
        )

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, data, **kwargs):
        # A^T * A
        return self.conv(data)

    def y_channel(self):
        return 3

    def to_pil(self, y):
        y = (y[0] + 1.0) / 2.0
        y = torch.clip(y, 0, 1)
        y = to_pil_image(y, "RGB")
        return y
