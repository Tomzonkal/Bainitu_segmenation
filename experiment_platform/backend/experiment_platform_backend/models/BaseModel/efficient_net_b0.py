import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import models
from models.utils import prepare_labels_int
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from models import ResNet50Model
from datasets import TorchImageDataset
import numpy as np
from tqdm import tqdm
import cv2

class EfficientNetB0(ResNet50Model):
    def __init__(self, input_dataset, epochs, lr=0.001, batch_size=32, shuffle=True, random_state=42):
        self.input_dataset = input_dataset
        self.epochs=epochs
        self.lr=lr
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V2)
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

        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V2)
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
