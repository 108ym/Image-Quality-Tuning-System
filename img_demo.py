import os
import torch
import cv2
import argparse
import warnings
import torchvision
import numpy as np
from utils import PSNR, validation, LossNetwork
from model.IAT_main import IAT
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from PIL import Image
import time  # Import the time module

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='demo_imgs/test.jpg')
parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--task', type=str, default='exposure')
config = parser.parse_args()

# Weights path
exposure_pretrain = r'best_Epoch.pth'

normalize_process = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

device = torch.device('cpu')

## Load Pre-train Weights
model = IAT().to(device)
if config.task == 'exposure':
    model.load_state_dict(torch.load(exposure_pretrain, map_location=device))
    
model.eval()

## Load Image
img = Image.open(config.file_name)
img = (np.asarray(img) / 255.0)
if img.shape[2] == 4:  # Remove alpha channel if present
    img = img[:, :, :3]
input = torch.from_numpy(img).float().to(device)
input = input.permute(2, 0, 1).unsqueeze(0)
if config.normalize:
    input = normalize_process(input)

## Forward Network
start_time = time.time()  # Start timing
_, _, enhanced_img = model(input)
end_time = time.time()  # End timing

# Calculate processing time
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.3f} seconds")

torchvision.utils.save_image(enhanced_img, 'result.png')
