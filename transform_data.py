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

def load_data(paths):
    img_array = []
    for path in paths:
        try:
            image = cv2.imread(path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
                img_array.append(image)
        except Exception as e:
            print(f'Failed to load {path}: {e}')
    return img_array

# load data
cars_real = load_data(real_images_paths)
cars_synth = load_data(fake_images_paths)


if __name__ == '__main__':

    # specify the target size for resizing
    target_size = (256, 256)

    # transforming the synthetic images once (resize + compression simulation)
    transformed_images = transform_images(cars_synth, fake_images_paths, target_size=target_size, compression=True, compress_to_jpeg=True, mask = True)
    images_no_transformation = transform_images(cars_synth, fake_images_paths, target_size=None, compression=False, compress_to_jpeg=True, mask = False)

    # transforming images 20 times, simulating online life
    def return_image_array():
        img = cars_synth
        for _ in range(20): 
            # resizing done once, as well as "converting" to jpeg via jpeg compression in the first step
            transformed_images = transform_images(img, fake_images_paths, target_size=target_size, compression=True, compress_to_jpeg=True, mask = False)
            img = transformed_images
            return img
    
    # display transformed images (showing first 5)
    plt.figure(figsize=(10,5))
    plt.text(0.5, 0.45, 'Resized and compressed images', ha='center', va='bottom', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.5, -0.05, 'Images with no resize, masking or compression', ha='center', va='top', fontsize=12, transform=plt.gca().transAxes)
    plt.axis('off')
    for i, img in enumerate(return_image_array()[:5]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
    for i, img in enumerate(images_no_transformation[:5]):
        plt.subplot(2, 5, i + 6)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
