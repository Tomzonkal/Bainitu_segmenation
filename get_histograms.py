import glob
import os
import cv2
import numpy as np

def get_histograms(segment_dir = "slic_out", bins=256):
    histograms = []
    labels = []
    segment_files = glob.glob(os.path.join(segment_dir, '*.png'))

    for seg_path in segment_files:
        filename = os.path.basename(seg_path)
        parts = filename.split('_')
        if len(parts) < 3:
            print(f"Skipping unrecognized filename format: {filename}")
            continue

        label_str = parts[2].lower()

        if label_str not in ['martensite', 'bainite']:
            print(f"Skipping unknown label '{label_str}' in file {filename}")
            continue

        # Map labels to integers TODO: error prone
        label = 1 if label_str == 'martensite' else 0

        img = cv2.imread(seg_path)
        if img is None:
            print(f"Could not read image: {seg_path}")
            continue

        # Calculate grayscale values on the fly
        gray_values = (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]).astype(np.uint8).flatten()

        # Compute histogram TODO: optimize bin number
        hist, _ = np.histogram(gray_values, bins=bins, range=(0, bins))

        # Normalize histogram
        hist = hist.astype('float32') / hist.sum()

        histograms.append(hist)
        labels.append(label)
    print(f"Loaded {len(histograms)} samples, bainite: {labels.count(0)}, martensite: {labels.count(1)} ")
    return histograms, labels
