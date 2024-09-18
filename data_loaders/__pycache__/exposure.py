import os
import os.path as osp

import torch
import torch.utils.data as data

import numpy as np
import glob
import random
import cv2
from glob import glob
import logging

random.seed(1143)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    filename='exposure.log', 
                    filemode='w',  # 'w' will overwrite the file every time the application starts
                    format='%(asctime)s - %(levelname)s - %(message)s')



def populate_train_list(images_path, mode='train'):
    train_list = [os.path.basename(f) for f in glob(os.path.join(images_path, '*.jpg'))]
    train_list.sort()

    if mode == 'train':
        random.shuffle(train_list)

    print(f"Found {len(train_list)} images in {images_path}")
    for img in train_list[:5]:  # Print first 5 images for debugging
        print(f"Image: {img}")

    return train_list

class exposure_loader(data.Dataset):

    def __init__(self, images_path, mode='train', expert='c', normalize=False):
        self.train_list = populate_train_list(images_path, mode)
        self.mode = mode  # train or test
        self.data_list = self.train_list
        self.low_path = images_path
        if self.mode == 'train' or self.mode == 'val':
            self.high_path = images_path.replace('INPUT_IMAGES', 'GT_IMAGES')
        elif self.mode == 'test':
            self.high_path = images_path.replace('INPUT_IMAGES', 'expert_'+expert+'_testing_set')
        self.normalize = normalize
        self.resize = True
        self.image_size = 640
        self.image_size_w = 480
        self.test_resize = True
        print("Total examples:", len(self.data_list))

    def FLIP_aug(self, low, high):
        if random.random() > 0.5:
            low = cv2.flip(low, 0)
            high = cv2.flip(high, 0)

        if random.random() > 0.5:
            low = cv2.flip(low, 1)
            high = cv2.flip(high, 1)

        return low, high
    
    def random_exposure(self,low):
        # Randomly choose an exposure level within the specified range and apply
        exposure_factor = np.random.uniform(-0.8, 1.2)
        image = cv2.convertScaleAbs(low, alpha=exposure_factor, beta=0)

        # Apply ±20% intensity variation
        intensity_factor = 1 + np.random.uniform(-0.2, 0.2)
        low = cv2.convertScaleAbs(low, alpha=intensity_factor, beta=0)
        return low


    def __getitem__(self, index):
        logging.debug(f"Fetching index {index}")
        img_id = self.data_list[index]

        # Extract the base ID and frame number according to your specific naming convention.
        base_id = img_id.split('_')[0]  # Assumes the ID is before the first underscore.
        frame_number = img_id.split('_')[-1].split('.')[0]  # Assumes frame number is right before ".jpg".

        # Construct paths for the low light and high light images.
        low_img_path = osp.join(self.low_path, img_id)
#         high_img_path = osp.join(self.high_path, f"{base_id}_processed_{frame_number}.jpg")
        logging.info(f"Loading low light image from: {low_img_path}")

        if self.mode in ['val', 'test']:
            high_img_path = osp.join(self.high_path, f"processed_{img_id}")
        else:  # 'train' mode or any other mode falls back to the default naming convention
            high_img_path = osp.join(self.high_path, f"{base_id}_processed_{frame_number}.jpg")
        
        logging.info(f"Loading high light image from: {high_img_path}")

        # Attempt to load images using OpenCV.
        try:
            data_lowlight = cv2.imread(low_img_path, cv2.IMREAD_UNCHANGED)
            data_highlight = cv2.imread(high_img_path, cv2.IMREAD_UNCHANGED)

            if data_lowlight is None or data_highlight is None:
                    raise ValueError(f"Failed to load one or both images at index {index}: {low_img_path}, {high_img_path}")

            # Check orientation and transpose if necessary.
            if data_lowlight.shape[0] >= data_lowlight.shape[1]:
                data_lowlight = cv2.transpose(data_lowlight)
                data_highlight = cv2.transpose(data_highlight)

            # Resize images to the specified dimensions.
            if self.resize:
                data_lowlight = cv2.resize(data_lowlight, (self.image_size, self.image_size_w))
                data_highlight = cv2.resize(data_highlight, (self.image_size, self.image_size_w))

            # Data augmentation for training mode.
            if self.mode == 'train':
                data_lowlight = self.random_exposure(data_lowlight)
                data_lowlight, data_highlight = self.FLIP_aug(data_lowlight, data_highlight)

            data_lowlight = np.asarray(data_lowlight[..., ::-1]) / 255.0
            data_highlight = np.asarray(data_highlight[..., ::-1]) / 255.0

            data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1)  # Convert to tensor and permute dimensions
            data_highlight = torch.from_numpy(data_highlight).float().permute(2, 0, 1)

            return data_lowlight, data_highlight, img_id

        except Exception as e:
            logging.error(f"Exception while processing images at index {index}: {str(e)}")
            return None


    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    train_path = '/Users/casseyyimei/Documents/GitHub/Illumination-Adaptive-Transformer/IAT_enhance/dataset/train/INPUT_IMAGES'
    train_dataset = exposure_loader(train_path, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1,
                                              pin_memory=True)
    for iteration, imgs in enumerate(train_loader):
        print(iteration)
        print(imgs[0].shape)
        print(imgs[1].shape)
        low_img = imgs[0]
        high_img = imgs[1]