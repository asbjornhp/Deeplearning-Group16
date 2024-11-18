import os 
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
def simulate_compression(image, image_type, noise_factor=0.05, blur_limit=3, brightness_limit=0.1, contrast_limit=0.1):
    if image_type in ['.jpg', '.jpeg']:
        # simulate JPEG compression (lossy) - add noise and blur
        compress = A.Compose([
            A.GaussianBlur(blur_limit=blur_limit, p=0.5),  # simulate blur from compression
            A.GaussNoise(var_limit=(0, noise_factor), p=0.5),  # add Gaussian noise
        ])
    elif image_type == '.png':
        # simulate PNG compression (lossless) - adjust brightness and contrast
        compress = A.RandomBrightnessContrast(brightness_limit,contrast_limit, p=0.5)  # slight brightness and contrast adjustments
    else:   
        return image # no compression for unsupported formats
    
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


def transform_images(image_list, image_paths, target_size=None, compression=False, mask = False):
    transformed_images = []
    # apply all transformations individually for each image
    for img, img_path in zip(image_list, image_paths):
        _, ext = os.path.splitext(img_path) # check the file extension to determine the image type
        ext = ext.lower() 

        transformed_image = img # original image

        if target_size: 
            transformed_image = resize_img(transformed_image, target_size, p=1.0)  # Apply resize with probability 1

        if compression: 
            transformed_image = simulate_compression(transformed_image, ext)

        if mask:
            transformed_image = apply_mask(transformed_image)

        transformed_images.append(transformed_image)  # add the transformed image to the list

    return transformed_images


