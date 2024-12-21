import subprocess
import os
import shutil
import argparse
from transformation import transform_images
from transform_data import load_data
from PIL import Image
import sys
import time
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Augmentation options')

parser.add_argument('--jpegcompress', action='store_true', help='Compress the images using jpeg compression')
parser.add_argument('--masking', action='store_true', help='Apply masking to the images')
parser.add_argument('--resize', action='store_true', help='Resize the images')
parser.add_argument('--batchSize', type=int, default=1, help='Batch size for the augmentation schema')
parser.add_argument('--jpegcompress_quality', type=int, default=95, help='Quality of the jpeg compression')

args = parser.parse_args()

print("Working directory: ", os.getcwd())

path = "weights/weights/univfd/fc_weights.pth"
print(f"Does path {path} exist: {os.path.exists(path)}")

print("The arguments are: ")
print("jpegcompress: ", args.jpegcompress)
print("masking: ", args.masking)
print("resize: ", args.resize)
print("batchSize: ", args.batchSize)
print("jpegcompress_quality: ", args.jpegcompress_quality)

# Original dataset folder
og_dataset_folder = "cars_dataset/cars"
aug_dataset_folder = f"cars_dataset_aug_{args.jpegcompress_quality}"
env_python = "../.venv_dl16_mpi/bin/python3" # Replace with actual python environment path

model_PatchCraft = [
    f"{env_python}", "evaluate.py",
    "--modelName=RPTC",
    "--ckpt=weights/weights/rptc/RPTC.pth",
    "--resultFolder=results/",
    f"--dataPath={aug_dataset_folder}"
]

model_UnivFD = [
    f"{env_python}", "evaluate.py", 
    "--modelName=UnivFD",
    "--ckpt=weights/weights/univfd/fc_weights.pth",
    "--resultFolder=results/",
    f"--dataPath={aug_dataset_folder}"
  ]

model_Rine = [
    f"{env_python}", "evaluate.py",
    "--modelName=Rine",
    "--ckpt=weights/weights/rine/model_4class_trainable.pth",
    "--resultFolder=results/",
    f"--dataPath={aug_dataset_folder}"
]

model_DeFake = [
    f"{env_python}", "evaluate.py",
    "--modelName=DeFake",
    "--ckpt=weights/weights/defake/clip_linear.pth",
    "--defakeClipEncoderPath=weights/weights/defake/finetune_clip.pt",
    "--defakeBlipPath=weights/weights/defake/model_base_capfilt_large.pth",
    "--resultFolder=results/",
    f"--dataPath={aug_dataset_folder}"
]

def process_single_batch(batch_file_paths, args):
    """
    Process a single batch of files.
    """
    images = load_data(batch_file_paths)  # Load images in the batch
    transformed_images = transform_images(
        images,
        quality_range=args.jpegcompress_quality,
        compression=args.jpegcompress,
        compress_to_jpeg=True,
        mask=args.masking,
    )
    save_transformed_images(batch_file_paths, transformed_images)


def save_transformed_images(file_paths, transformed_images):
    """
    Save transformed images back to their original paths.
    """
    for file_path, transformed_image in zip(file_paths, transformed_images):
        #if "1_fake" in file_path:
        #    extension = ".png"
        #else:
        #    extension = ".jpeg"
        extension = ".png"
        
        # Remove duplicate extensions
        #file_path = file_path.rsplit(".jpeg", 1)[0]
        file_path = file_path.rsplit(".png", 1)[0]

        transformed_img = Image.fromarray(transformed_image)  # Convert numpy array back to PIL Image
        
        #if extension == ".jpeg":
        #    transformed_img.save(file_path + extension, quality='keep')
        #else:
        transformed_img.save(file_path + extension)

def copy_and_rename_jpg_to_png(src_folder, dest_folder):
    """
    Copies the folder structure from src_folder to dest_folder,
    renaming all files with .jpg extension to .png.
    """
    for root, dirs, files in os.walk(src_folder):
        # Compute the destination path
        relative_path = os.path.relpath(root, src_folder)
        dest_path = os.path.join(dest_folder, relative_path)
        
        # Ensure the destination path exists
        os.makedirs(dest_path, exist_ok=True)
        
        for file in files:
            src_file_path = os.path.join(root, file)
            
            if file.endswith(".jpg"):
                # Rename .jpg to .png
                new_file_name = file.rsplit(".jpg", 1)[0] + ".png"
            else:
                # Keep the original name for other files
                new_file_name = file
            
            dest_file_path = os.path.join(dest_path, new_file_name)
            
            # Copy the file to the new location
            shutil.copy2(src_file_path, dest_file_path)

for i in range(43):
    start = time.time()
    if i == 0:
        
        # Copy the original folder and place the copied images in the new folder, 
        #following the same structure as the original folder

        # copy the original dataset to the new folder using shutil
        if os.path.exists(aug_dataset_folder):
            shutil.rmtree(aug_dataset_folder)
        copy_and_rename_jpg_to_png(og_dataset_folder, aug_dataset_folder)

        subprocess.run(model_PatchCraft + [f"--jpegCompression={args.jpegcompress_quality}", f"--transformIterations={i}"], text=True)
        subprocess.run(model_UnivFD + [f"--jpegCompression={args.jpegcompress_quality}", f"--transformIterations={i}"], text=True)
        subprocess.run(model_Rine + [f"--jpegCompression={args.jpegcompress_quality}", f"--transformIterations={i}"], text=True)
        subprocess.run(model_DeFake + [f"--jpegCompression={args.jpegcompress_quality}", f"--transformIterations={i}"], text=True)
        end = time.time()
    else:
        # Augment the augmented folder and place the augmented images in the new folder, following the same structure as the original folder
        # > Run augmentation on the augmented dataset
        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(aug_dataset_folder)
            for file in files if file.endswith(".jpg") or file.endswith(".png")
        ]
        
        batches = [
            all_files[j:j + args.batchSize]
            for j in range(0, len(all_files), args.batchSize)
        ]
        print("Processing batches", file=sys.stdout)
        # Use multiprocessing to process batches in parallel
        with Pool() as pool:
            pool.starmap(process_single_batch, [(batch, args) for batch in batches])
        print("Finished processing batches", file=sys.stdout)
        subprocess.run(model_PatchCraft + [f"--jpegCompression={args.jpegcompress_quality}", f"--transformIterations={i}"])
        subprocess.run(model_UnivFD + [f"--jpegCompression={args.jpegcompress_quality}", f"--transformIterations={i}"])
        subprocess.run(model_Rine + [f"--jpegCompression={args.jpegcompress_quality}", f"--transformIterations={i}"])
        subprocess.run(model_DeFake + [f"--jpegCompression={args.jpegcompress_quality}", f"--transformIterations={i}"])

    end = time.time()
    
    print(f"Time taken for iteration {i}: {end - start}", file=sys.stdout)