from transformation import transform_images
import cv2
import matplotlib.pyplot as plt
import os 

def load_image_paths(base_path):
    real_image_paths = []
    synth_image_paths = []

    # define the subfolders for real and fake images
    real_folder = '0_real'
    fake_folder = '1_fake'

    # construct full paths for real and fake folders
    real_folder_path = os.path.join(base_path, 'cars', real_folder)
    synth_folder_path = os.path.join(base_path, 'cars', fake_folder)

    # load images from the '1_real' folder
    if os.path.exists(real_folder_path) and os.path.isdir(real_folder_path):
        for file_name in os.listdir(real_folder_path):
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):  # filter image files
                real_image_paths.append(os.path.join(real_folder_path, file_name))
    
    # load images from the '0_fake' folder
    if os.path.exists(synth_folder_path) and os.path.isdir(synth_folder_path):
        for file_name in os.listdir(synth_folder_path):
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):  # filter image files
                synth_image_paths.append(os.path.join(synth_folder_path, file_name))

    return real_image_paths, synth_image_paths

base_path = 'cars_dataset'
real_images_paths, synth_images_paths = load_image_paths(base_path)

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
real_images = load_data(real_images_paths)
synth_images = load_data(synth_images_paths)


if __name__ == '__main__':

    # specify the target size for resizing
    target_size = (256, 256)

    # transforming the synthetic images once (resize + compression simulation)
    transformed_images = transform_images(synth_images, quality_range = 95, target_size=target_size, compression=True, compress_to_jpeg=True, mask = True)
    images_no_transformation = transform_images(synth_images, quality_range = 95, target_size=None, compression=False, compress_to_jpeg=True, mask = False)

    # transforming images 20 times, simulating online life
    def return_image_array():
        img = synth_images
        for _ in range(20): 
            # resizing done once, as well as "converting" to jpeg via jpeg compression in the first step
            transformed_images = transform_images(img,quality_range = 95, target_size=target_size, compression=True, compress_to_jpeg=True, mask = True)
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
