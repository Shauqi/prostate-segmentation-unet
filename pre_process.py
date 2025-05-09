import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_nii_files(directory):
    # check if the directory exists
    if not os.path.exists(directory):
        raise Exception('The directory does not exist')

    # check if the directory is empty
    if not os.listdir(directory):
        raise Exception('The directory is empty')

    # check if the directory contains 6 subdirectories
    subdirectories = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
    if not all([os.path.exists(os.path.join(directory, subdirectory)) for subdirectory in subdirectories]):
        raise Exception('The directory does not contain the 6 subdirectories')

    # load .nii files from the subdirectories
    nii_files = []
    for subdirectory in subdirectories:
        for file in os.listdir(os.path.join(directory, subdirectory)):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nii_files.append(os.path.join(directory, subdirectory, file))

    # sort the .nii files
    nii_files.sort()
    
    return nii_files


# test the function
# directory = 'Processed_data_nii'
directory = 'input'
nii_files = load_nii_files(directory)


print(nii_files)
print(len(nii_files))


counter = 0
for i in range(0, len(nii_files), 2):
    for depth in range(nib.load(nii_files[i]).shape[2]):
        if nib.load(nii_files[i+1]).get_fdata()[:, :, depth].max() != 0:
            counter += 1
            # retrieve the image and mask
            img = nib.load(nii_files[i]).get_fdata()[:, :, depth]
            msk = nib.load(nii_files[i+1]).get_fdata()[:, :, depth] >= 1
            # save image into data/img
            plt.imsave(f'./data/img/{counter}.png', img, cmap='gray')
            # save mask into data/mask
            plt.imsave(f'./data/mask/{counter}.png', msk, cmap='gray')