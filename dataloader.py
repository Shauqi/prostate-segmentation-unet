import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ProstateDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith('.png')])
        self.masks = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale using pillow library
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale (binary mask) 'L' stands for "luminance" 

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample