import os
import random
from PIL import Image, ImageSequence
import cv2
import numpy as np


def save_tif_frames_to_jpg(input_directory, output_train_directory, output_val_directory, train_ratio=0.7):
    # Ensure output directories exist
    os.makedirs(output_train_directory, exist_ok=True)
    os.makedirs(output_val_directory, exist_ok=True)

    # Gather all TIFF files
    files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.tif', '.tiff'))]
    random.shuffle(files)  # Shuffle to randomize the distribution

    # Split files based on the training ratio
    split_index = int(len(files) * train_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    # Function to process and save images
    def process_and_save(files, output_directory):
        for filename in files:
            file_path = os.path.join(input_directory, filename)
            try:
                with Image.open(file_path) as img:
                    for i, frame in enumerate(ImageSequence.Iterator(img)):
                        # Convert to RGB if image is in palette mode
                        if frame.mode == 'P':
                            frame = frame.convert('RGB')
                        
                        output_filename = f"{os.path.splitext(filename)[0]}_frame_{i}.jpg"
                        output_path = os.path.join(output_directory, output_filename)
                        frame.save(output_path, 'JPEG')
                        print(f"Saved {output_path}")  # Log each saved frame
            except IOError as e:
                print(f"Error opening or processing file {file_path}: {e}")
            except Exception as e:
                print(f"Unexpected error with file {file_path}: {e}")

    # Process and save training and validation images
    process_and_save(train_files, output_train_directory)
    process_and_save(val_files, output_val_directory)


input_dir = '/Users/casseyyimei/Documents/GitHub/Illumination-Adaptive-Transformer/IAT_enhance/dataset/train/overkill'
output_train_dir = '/Users/casseyyimei/Documents/GitHub/Illumination-Adaptive-Transformer/IAT_enhance/dataset/train/INPUT_IMAGES'
output_val_dir = '/Users/casseyyimei/Documents/GitHub/Illumination-Adaptive-Transformer/IAT_enhance/dataset/validation/INPUT_IMAGES'
save_tif_frames_to_jpg(input_dir, output_train_dir, output_val_dir)


def enhance_and_denoise_images(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Process each image file in the directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('.jpg'):
            
            input_path = os.path.join(input_directory, filename)
            output_filename = f"processed_{filename}"  
            output_path = os.path.join(output_directory, output_filename)

            # Open the image and enhance contrast
            image = Image.open(input_path)
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(1.5)  # You can adjust the factor as needed

            # Convert enhanced image from PIL to OpenCV format
            enhanced_image_cv = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
            
            # Apply denoising using OpenCV
            denoised_image_cv = cv2.fastNlMeansDenoisingColored(enhanced_image_cv, None, 9, 7, 7)
            
            # Convert back to PIL image for any further processing or saving
            final_image = Image.fromarray(cv2.cvtColor(denoised_image_cv, cv2.COLOR_BGR2RGB))

            # Save the processed image
            final_image.save(output_path)
            print(f"Processed and saved image: {output_path}")


# 
# input_directory = '/Users/casseyyimei/Documents/GitHub/Illumination-Adaptive-Transformer/IAT_enhance/report_img'  # Path to your input directory containing JPEG images
# output_directory = '/Users/casseyyimei/Documents/GitHub/Illumination-Adaptive-Transformer/IAT_enhance/report_img'  # Path to save the enhanced and denoised images
# enhance_and_denoise_images(input_directory, output_directory)

