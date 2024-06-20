import torch
import numpy as np

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, var=1.0, p=0.3):
        super().__init__()
        self.mean = mean
        self.var = var
        self.p = p

    def forward(self, img):
        if np.random.random() > self.p:
            return img
        im =  img + torch.from_numpy(np.random.normal(loc=self.mean, scale=self.var, size=img.shape)).to(img.dtype)
        return torch.clip(im, 0, 1)