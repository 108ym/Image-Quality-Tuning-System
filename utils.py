import torch
import torch.nn as nn
import numpy as np
from math import exp

from IQA_pytorch import SSIM
import matplotlib.pyplot as plt

# calculate PSNR
class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))


ssim = SSIM()
psnr = PSNR()

def validation(model, val_loader):

    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            _, _, enhanced_img = model(low_img)
            # print(enhanced_img.shape)
        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        #ssim_value = ssim(enhanced_img, high_img).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        # print('The %d image SSIM value is %d:' %(i, ssim_value))
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
    return SSIM_mean, PSNR_mean

def validation_shadow(model, val_loader):

    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img, mask = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda()
            _, _, enhanced_img = model(low_img, mask)
            # print(enhanced_img.shape)
        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        #ssim_value = ssim(enhanced_img, high_img).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        # print('The %d image SSIM value is %d:' %(i, ssim_value))
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
    return SSIM_mean, PSNR_mean



