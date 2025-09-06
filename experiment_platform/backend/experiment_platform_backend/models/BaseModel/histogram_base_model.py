import re
import numpy as np
import cv2
import os
from models.utils import get_histograms, prepare_labels
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class HistogramBaseModel:
    """
    Base class for all models in the experiment platform.
    This class provides common functionality that can be shared across different model implementations.
    """

    def __init__(self, input_dataset, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,bins,n_splits=5, shuffle=True, random_state=42):
        self.input_dataset = input_dataset
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.bins = bins
        self.model = None

    def _create_histograms(self, segments_paths):
        """
        Create histograms for the given segments.
        :param segments_paths: List of paths to segment images.
        :return: List of histograms and corresponding labels.
        """
        histograms = []
        
        for seg_path in segments_paths:
            image = cv2.imread(seg_path)
            if image is None:
                print(f"Could not read image: {seg_path}")
                continue
            hist= get_histograms(image, bins=self.bins)
            histograms.append(hist)
        return histograms
            
    def prepare_X(self):
        """
        Prepare the feature matrix X from the histograms.
        :param histograms: List of histograms.
        :return: Numpy array of features.
        """
        input_dataset = self.input_dataset.load_meta_data()
        segments_paths= input_dataset['image_path'].tolist()
        histograms = self._create_histograms(segments_paths)
        X = np.array(histograms)
        return X

    def prepare_y(self):
        """
        Prepare the target vector y from the labels.
        :param labels: List of labels.
        :return: Numpy array of targets.
        """
        input_dataset = self.input_dataset.load_meta_data()
        labels = input_dataset['json_path']
        y = prepare_labels(labels)
        return y

    def _validate(self, y, y_predict):
        """
        Validate the model using cross-validation.
        :param X: Feature matrix.
        :param y: Target vector.
        :return: Validation metrics.
        """
        # Calculate metrics on combined predictions from all folds
        acc = accuracy_score(y, y_predict)
        prec = precision_score(y, y_predict, average='weighted', zero_division=0)
        rec = recall_score(y, y_predict, average='weighted', zero_division=0)
        f1 = f1_score(y, y_predict, average='weighted', zero_division=0)
        labels = np.unique(y)
        cm = confusion_matrix(y, y_predict  , labels=labels)
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        return metrics, cm, labels

    def train(self, X, y):
        """
        Train the model using the prepared data.
        :return: Trained model.
        """ 

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metric_dict={}
        y_true_all = []
        y_pred_all = []

        for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            print(f"Processing {i} fold...")
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            
            metric_dict[f"set_{i}"],_,_=self._validate(y=y_valid,y_predict=y_pred)

            y_true_all.extend(y_valid)
            y_pred_all.extend(y_pred)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        model=RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )
        self.model = model.fit(X, y)
        metric_dict["avg_metric"], cm, labels=self._validate(y_true_all, y_pred_all)    
        return metric_dict, cm, labels          


