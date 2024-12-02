
"""
This script calculates the Radially Averaged Power Spectra (RAPSD) and the regular Power Spectra of input images.
The values can then be averaged for plotting.
"""

# Importing packages
import numpy as np
import cv2

# RAPSD and Power Spectrum

def spectra(image_paths):

    # Load images
    image_list = []
    for path in image_paths:
        try:
            image = cv2.imread(path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
                image_list.append(image)
        except Exception as e:
            print(f'Failed to load {path}: {e}')

    rapsd = []
    power_spectra = []

    for k in range(len(image_list)): 

        image_data = image_list[k]

        for channel in range(image_data.shape[2]):  # Loop over RGB channels
            channel_data = image_data[..., channel]

            # Compute power spectrum for this channel
            f_transform = np.fft.fft2(channel_data)
            f_transform_shifted = np.fft.fftshift(f_transform)
            pow_spec = np.abs(f_transform_shifted) ** 2 

            
        # Computing the power spectrum
        spec = np.fft.fftshift(np.fft.fft2(channel_data))
        spec = np.abs(spec)
        spec = np.log(spec + 1)
        power_spectra.append(spec)
        

        # Creating a radial grid
        ny, nx = pow_spec.shape
        y, x = np.indices((ny, nx))
        center_y, center_x = ny // 2, nx // 2
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)


        # Radial binning 
        r = r.astype(np.int32)
        psd = np.bincount(r.ravel(), weights=pow_spec.ravel()) / np.bincount(r.ravel())
        rapsd.append(psd)    


    return rapsd, power_spectra


# Averaging

def average_spectra(spectra_list):
    avg_spectrum = np.mean(spectra_list, axis=0)
    return avg_spectrum





