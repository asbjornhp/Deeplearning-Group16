
"""
Peak Signal-to-Noise Ratio (PSNR) is a quality metric that can be used to compare the quality of two images.
The higher the PSNR, the better the quality of the image.
This script calculates the PSNR between the uncompressed and compressed synthetic images.

# Needs to be called separately for real and fake images. 
"""

# Importing packages
import os
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize


def psnr_calculation(transformed_path, original_path):
    """
    Calculate PSNR values between transformed and original images.

    Parameters:
        transformed_path (str): Path to the directory containing transformed images.
        original_path (str): Path to the directory containing original images.

    Returns:
        list: PSNR values for each pair of images.
    """
    # Get sorted lists of image file names from both directories
    transformed_files = sorted(os.listdir(transformed_path))
    original_files = sorted(os.listdir(original_path))
    
    psnr_values = []  # To store PSNR values at each iteration

    # Compute PSNR for each pair of images
    for t_file, o_file in zip(transformed_files, original_files):
        # Load the images
        transformed_image = imread(os.path.join(transformed_path, t_file))
        original_image = imread(os.path.join(original_path, o_file))

        # Resize original image to match transformed image shape
        reshaped_original = resize(original_image, transformed_image.shape, anti_aliasing=True, preserve_range=True)

        # Calculate PSNR
        psnr_value = psnr(reshaped_original, transformed_image, data_range=reshaped_original.max() - reshaped_original.min())
        psnr_values.append(psnr_value)

    return psnr_values



