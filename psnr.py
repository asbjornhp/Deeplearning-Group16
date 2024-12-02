
"""
Peak Signal-to-Noise Ratio (PSNR) is a quality metric that can be used to compare the quality of two images.
The higher the PSNR, the better the quality of the image.
This script calculates the PSNR between the uncompressed and compressed synthetic images.
"""

# Importing packages from skimage:
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize


def psnr_calculation(transformed_list, original_list):

    img = transformed_list                # Transformed input image
    org = original_list                   # Original input image (not transformed)
    psnr_values = []                      # To store PSNR values at each iteration

    # Compute PSNR for each image
    for j in range(len(img)):
        # Resize original image to match transformed image shape
        reshaped_original = resize(org[j], img[j].shape, anti_aliasing=True, preserve_range=True)

        # Calculate PSNR
        psnr_value = psnr(reshaped_original, img[j], data_range=reshaped_original.max() - reshaped_original.min())
        psnr_values.append(psnr_value)
    
  
    return psnr_values


