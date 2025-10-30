from torch.utils.data import Dataset
import torch
import cv2

class TorchImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            # Convert BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(label).long()  # Convert label to tensor
        return img, label

    def __len__(self):
        return len(self.images)
