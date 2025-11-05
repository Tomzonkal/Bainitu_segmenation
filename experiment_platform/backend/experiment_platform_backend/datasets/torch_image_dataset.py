from torch.utils.data import Dataset
import torch
import cv2

class TorchImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, grayscale_conversion=False):
        """
        Args:
            images (list or np.ndarray): List/array of image data (as NumPy arrays).
            labels (list or np.ndarray): Corresponding labels.
            transform (callable, optional): Optional transform to apply on an image.
            grayscale_conversion (bool): If True, converts images to grayscale.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.grayscale_conversion = grayscale_conversion

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # --- Optional grayscale conversion ---
        if self.grayscale_conversion:
            if len(img.shape) == 3:  # Only convert if it's color
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Apply transform if provided ---
        if self.transform:
            img = self.transform(img)
        else:
            if self.grayscale_conversion:
                # Add single channel dimension for grayscale
                img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
            else:
                # Convert BGR → RGB and reorder to [C, H, W]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(label).long()
        return img, label

    def __len__(self):
        return len(self.images)
