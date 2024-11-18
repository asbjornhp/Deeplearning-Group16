from transformation import transform_images
import cv2
import matplotlib.pyplot as plt
import os 


# function to get all image file paths from subfolders
def load_image_paths(base_path):
    image_paths = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):  # check if it's a folder
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):  # filter image files
                    image_paths.append(os.path.join(folder_path, file_name))
    return image_paths

real_images_paths = load_image_paths('cars_real')
fake_images_paths = load_image_paths('cars_fake')

# number of total images
print(f'Total real images:\t{len(real_images_paths)}') # 64
print(f'Total fake images:\t{len(fake_images_paths)}') # 64

# load data
cars_real = []
for carpath in real_images_paths:
    try:
        image = cv2.imread(carpath)
        if image is not None:  # verify loading was successful
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
            cars_real.append(image)
    except Exception as e:
        print(f'Failed to load {carpath}: {e}')

cars_synth = []
for carpath in fake_images_paths:
    try:
        image = cv2.imread(carpath)
        if image is not None: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            cars_synth.append(image)
    except Exception as e:
        print(f'Failed to load {carpath}: {e}')


# specify the target size for resizing
target_size = (256, 256)

# transform the images (resize + compression simulation)
transformed_images_resized = transform_images(cars_synth, fake_images_paths, target_size=target_size, compression=True, mask = True)
transformed_images_no_resize = transform_images(cars_synth, fake_images_paths, target_size=None, compression=False, mask = False)

# display transformed images (showing first 5)
plt.figure(figsize=(10,5))
plt.text(0.5, 0.45, 'Resized, masked and compressed images', ha='center', va='bottom', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.5, -0.05, 'Images with no resize, masking or compression', ha='center', va='top', fontsize=12, transform=plt.gca().transAxes)
plt.axis('off')
for i, img in enumerate(transformed_images_resized[:5]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
for i, img in enumerate(transformed_images_no_resize[:5]):
    plt.subplot(2, 5, i + 6)
    plt.imshow(img)
    plt.axis('off')
plt.show()
