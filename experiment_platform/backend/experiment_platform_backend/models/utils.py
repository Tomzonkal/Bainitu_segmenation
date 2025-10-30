import glob
import os
import re
import cv2
import numpy as np
import json
def get_histograms(image, bins=256):
        # Calculate grayscale values on the fly
        gray_values = (0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]).astype(np.uint8).flatten()
        # TODO: what was the idea with range=0,bins?? Should it stay?
        hist, _ = np.histogram(gray_values, bins=bins) #, range=(0,bins))

        # Normalize histogram
        hist = hist.astype('float32') / hist.sum()

        return hist
    
    
def prepare_labels(metadata_paths):
    """
    Prepare labels from the input dataset.
    
    :param input_dataset: Dataset object containing image paths and labels.
    :return: Numpy array of labels.
    """
    labels = []
    for path in metadata_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file {path} does not exist.")
        with open(path, 'r') as f:
            data = json.load(f)
            labels.append(data['label'])
            
    return np.array(labels)

def prepare_labels_int(metadata_paths):
    labels = []
    for path in metadata_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file {path} does not exist.")
        with open(path, 'r') as f:
            data = json.load(f)
            labels.append(data['label'])

    # Map string labels to integer indices
    classes = sorted(list(set(labels)))             # unique labels
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in labels], dtype=np.int64)
    
    return y, class_to_idx