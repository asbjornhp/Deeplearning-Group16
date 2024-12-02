from transformation_single_img import transform_image
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

    augmented_images = []
    for image in synth_images: # augment images once       
        # transforming the synthetic images once (resize + compression simulation)
        augmented_image = transform_image(image, quality_range = 95, target_size=target_size, compression=True, compress_to_jpeg=True, mask = True)
        augmented_images.append(augmented_image)

    # plotting the 5 synthetic images and 5 augmented synthetic images
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows and 5 columns for 10 images
    axes = axes.flatten()

    # display 5 synthetic images
    for i, synth_image in enumerate(synth_images[:5]):  
        axes[i].imshow(synth_image)
        axes[i].axis('off') 
        axes[i].set_title(f'Synthetic {i+1}')

    # display 5 augmented synthetic images
    for i, augmented_image in enumerate(augmented_images[:5]):  
        axes[i+5].imshow(augmented_image)
        axes[i+5].axis('off')  
        axes[i+5].set_title(f'Augmented synthetic {i+1}')

    # show the plot
    plt.tight_layout()
    plt.show()
