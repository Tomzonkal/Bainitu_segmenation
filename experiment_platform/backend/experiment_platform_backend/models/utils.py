import glob
import os
import re
import cv2
import numpy as np

def get_histograms(image, bins=256):
        # Calculate grayscale values on the fly
        gray_values = (0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]).astype(np.uint8).flatten()

        # Compute histogram TODO: optimize bin number
        hist, _ = np.histogram(gray_values, bins=bins, range=(0, bins))

        # Normalize histogram
        hist = hist.astype('float32') / hist.sum()

        return hist