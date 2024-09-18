import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import argparse
import numpy as np
from utils import PSNR
from model.IAT_main import IAT
from IQA_pytorch import SSIM
from data_loaders.exposure import exposure_loader
from tqdm import tqdm
import time  

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--img_val_path', type=str, default="Illumination-Adaptive-Transformer/IAT_enhance/dataset/test/INPUT_IMAGES")
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--expert', type=str, default='a')  
parser.add_argument('--pre_norm', type=bool, default=False) 
config = parser.parse_args()

print(config)
test_dataset = exposure_loader(images_path=config.img_val_path, mode='test',  expert=config.expert, normalize=config.pre_norm)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = IAT(type='exp').cuda()
model.load_state_dict(torch.load("Illumination-Adaptive-Transformer/IAT_enhance/best_Epoch.pth"))
model.eval()
            


ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []
time_list = [] 


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_val_path.replace('INPUT_IMAGES', 'Result')
    mkdir(result_path)

with torch.no_grad():
    for i, imgs in tqdm(enumerate(test_loader)):
        start_time = time.time()
        #print(i)
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        #print(low_img.shape)
        mul, add ,enhanced_img = model(low_img)
        
        end_time = time.time()  # End timing
        inference_time = end_time - start_time
        time_list.append(inference_time)  # Append the test time

        if config.save:
            img_id = imgs[2][0].split('.')[0]
            save_path = os.path.join(result_path, f'{img_id}.png') 
            torchvision.utils.save_image(enhanced_img, save_path)

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)


SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
mean_inference_time = np.mean(time_list)  # Calculate mean test time
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)
print('The Mean Inference Time is:', mean_inference_time)  

