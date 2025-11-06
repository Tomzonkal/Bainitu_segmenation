import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from models import ResNet50Model
from datasets import TorchImageDataset

class CustomGrayCNN(ResNet50Model):
    def __init__(self, input_dataset, epochs=20, lr=0.001, batch_size=32, shuffle=True, random_state=42):
        self.input_dataset = input_dataset
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        print("Using device:", self.device)

def _build_model(self, num_classes, input_size=(224, 224)):
    # Convolutional backbone
    conv_layers = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    # Compute flattened feature size dynamically
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *input_size)  # 1 channel for grayscale
        dummy = conv_layers(dummy)
        flattened_size = dummy.numel()

    fc_layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flattened_size, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    return nn.Sequential(conv_layers, fc_layers)

    def _validate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        labels = np.unique(y_true)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, cm, labels

    def train(self, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        metric_dict = {}
        y_true_all, y_pred_all = [], []

        dataset = TorchImageDataset(X, y, grayscale_conversion=True)

        for i, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            print(f"\n---- Fold {i + 1}/5 ----")

            train_subset = Subset(dataset, train_idx)
            valid_subset = Subset(dataset, valid_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False)

            model = self._build_model(num_classes=len(np.unique(y))).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)

            # --- Training ---
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

        # --- Average Metrics ---
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        metric_dict["avg_metric"], cm, labels = self._validate(y_true_all, y_pred_all)
        return metric_dict, cm, labels
