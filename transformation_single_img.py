import albumentations as A
import numpy as np
from random import randint
import cv2

def resize_img(image, shape, p=1.0):
    x, y = shape
    resize = A.Resize(height=x, width=y, p=p)  # pass height and width separately

    augmented = resize(image=image)
    return augmented['image']

# compression function based on image type (assuming only JPEG or PNG)
def simulate_compression(image, quality_range, compress_to_jpeg=True):
    if compress_to_jpeg:
        # JPEG compression (lossy) 
        compress = A.ImageCompression(quality_range, compression_type='jpeg', p=1.0) # compression quality of 95%
    else:
        # PNG compression (lossless) - using simulated lossless webp compression 
        compress = A.ImageCompression(quality_range, compression_type='webp', p=1.0)  
    augmented = compress(image=image)
    return augmented['image']
  

def apply_mask(image, mask_color=(0, 0, 0), precentage_masking = 0.1):
    h, w, _ = image.shape
    p = precentage_masking
    mask_size=(p*h, p*w) # mask size dependent on size of image, to reduce loss of information/ identifiable threshold
    mask = np.ones_like(image) * 255  
    x = randint(0, round(w - mask_size[0])) # location
    y = randint(0, round(h - mask_size[1]))
    mask[y:round(y + mask_size[1]), x:round(x + mask_size[0])] = mask_color

    masked_image = cv2.bitwise_and(image, mask)   
    return masked_image

def transform_image(img, quality_range = 95, target_size = None, compression = True, compress_to_jpeg = True, mask = False):

    augmented_img = img # original image

    if target_size: 
        augmented_img = resize_img(augmented_img, target_size)  

    if compression: 
        augmented_img = simulate_compression(augmented_img, quality_range, compress_to_jpeg)

    if mask:
        augmented_img = apply_mask(augmented_img)

    return augmented_img
