import os
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

input_real = 'cars_real'
output_real = 'Jpeg_real'
input_fake = 'cars_fake'
output_fake = 'Jpeg_fake'

# JPEG compression quality
jpeg_quality = 100

# Function to process dataset and convert PNG to JPEG
def process_dataset_with_conversion(input_dir, output_dir):
    dataset = datasets.ImageFolder(root=input_dir, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Total number of images in {input_dir}: {len(dataset)}")

    total_images = 0
    for idx, (images, labels) in enumerate(data_loader):
        for i in range(images.size(0)):  # Loop through the batch (batch size = 1 here)
            pil_image = transforms.ToPILImage()(images[i])

            # Define the output path and change extension to .jpg
            original_file_path = dataset.imgs[total_images][0]
            original_file_name = os.path.splitext(os.path.basename(original_file_path))[0]  # Get file name without extension
            output_image_path = os.path.join(output_dir, f"{original_file_name}.jpg")

            # Save with JPEG compression
            pil_image = pil_image.convert("RGB")  # Ensure no transparency for PNGs
            pil_image.save(output_image_path, 'JPEG', quality=jpeg_quality)

            # File size comparison
            original_size = os.path.getsize(original_file_path)
            compressed_size = os.path.getsize(output_image_path)
            print(f"Image: {os.path.basename(output_image_path)}")
            print(f"Original Size: {original_size} bytes")
            print(f"Compressed Size: {compressed_size} bytes")
            print(f"Compression Ratio: {compressed_size / original_size:.2%}\n")

            total_images += 1

        if total_images % 64 == 0:
            print(f"Processed {total_images}/{len(dataset)} images")

    print(f"JPEG conversion and compression completed for {input_dir}!")

process_dataset_with_conversion(input_real, output_real)
process_dataset_with_conversion(input_fake, output_fake)
