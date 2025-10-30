import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import models
from models.utils import prepare_labels_int
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from  models import HistogramBaseModel
from datasets import TorchImageDataset
import numpy as np
from tqdm import tqdm
import cv2

class ResNet50Model(HistogramBaseModel):
    def __init__(self, input_dataset, epochs, lr=0.001, batch_size=32, shuffle=True, random_state=42):
        self.input_dataset = input_dataset
        self.epochs=epochs
        self.lr=lr
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_max_image_size(self, image_paths):
        """Compute max width and height across all images."""
        max_w, max_h = 0, 0
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Could not read image: {path}")
                continue
            h, w = img.shape[:2]
            max_w = max(max_w, w)
            max_h = max(max_h, h)
        return max_w, max_h

    def pad_image_to_size(self, image, target_w, target_h, pad_value=(0, 0, 0)):
        """Pad an image symmetrically to target width and height."""
        h, w = image.shape[:2]
        top = (target_h - h) // 2
        bottom = target_h - h - top
        left = (target_w - w) // 2
        right = target_w - w - left
        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=pad_value
        )
        return padded

    def prepare_X(self, target_size=(224, 224), pad_to_largest=False):
        """Resize or pad all segment images to a common size."""
        input_dataset = self.input_dataset.load_meta_data()
        segments_paths= input_dataset['image_path'].tolist()
        processed_images = []

        if pad_to_largest:
            # compute max image size first
            max_w, max_h = self.get_max_image_size(segments_paths)
            print(f"Padded to largest image size: ({max_w}, {max_h})")
        else:
            max_w, max_h = target_size

        for seg_path in segments_paths:
            image = cv2.imread(seg_path)
            if image is None:
                print(f"Could not read image: {seg_path}")
                continue

            if pad_to_largest:
                padded = self.pad_image_to_size(image, max_w, max_h)
                processed_images.append(padded)
            else:
                resized = cv2.resize(image, (max_w, max_h), interpolation=cv2.INTER_AREA)
                processed_images.append(resized)

        return processed_images

    def prepare_y(self):
        """
        Prepare the target vector y from the labels.
        :param labels: List of labels.
        :return: Numpy array of targets.
        """
        input_dataset = self.input_dataset.load_meta_data()
        labels = input_dataset['json_path']
        y, class_to_idx = prepare_labels_int(labels)
        return y

    def train(self, X, y):
        """
        dataset: torchvision Dataset containing (image, label) pairs
        y: numpy array of labels for StratifiedKFold
        """
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metric_dict = {}
        y_true_all, y_pred_all = [], []
        
        dataset = TorchImageDataset(X, y)

        for i, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            print(f"\n---- Fold {i + 1}/5 ----")

            # Create subset dataloaders
            train_subset = Subset(dataset, train_idx)
            valid_subset = Subset(dataset, valid_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False)

            # New model each fold (reset)
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, len(np.unique(y)))
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.fc.parameters(), lr=self.lr)

            # --- Training Loop ---
            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                print(f"Fold {i+1}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

            # --- Validation ---
            model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for imgs, labels in valid_loader:
                    imgs = imgs.to(self.device)
                    outputs = model(imgs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    y_pred.extend(preds)
                    y_true.extend(labels.numpy())

            fold_metrics, _, _ = self._validate(y_true, y_pred)
            metric_dict[f"fold_{i+1}"] = fold_metrics
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

        # Train final model on all data
        print("\nTraining final model on full dataset...")
        final_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, len(np.unique(y)))
        self.model = self.model.to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            for imgs, labels in tqdm(final_loader, desc=f"Final Train Epoch {epoch+1}/{self.epochs}", leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Compute final averaged metrics
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        metric_dict["avg_metric"], cm, labels = self._validate(y_true_all, y_pred_all)
        return metric_dict, cm, labels
