import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

segment_dir = 'output_segments'

# Initialize dictionaries to hold histograms per category
histograms = {
    'martensite': [],
    'bainite': []
}

# Get all segment images
segment_files = glob.glob(os.path.join(segment_dir, '*.png'))

for seg_path in segment_files:
    filename = os.path.basename(seg_path)
    # Assuming filename pattern: base_label_index.png
    # Extract label by splitting on underscores
    parts = filename.split('_')
    if len(parts) < 3:
        print(f"Skipping unrecognized filename format: {filename}")
        continue

    label = parts[-2].lower()  # second last part is label

    if label not in histograms:
        print(f"Skipping unknown label '{label}' in file {filename}")
        continue

    # Read image
    img = cv2.imread(seg_path)
    if img is None:
        print(f"Could not read image: {seg_path}")
        continue

    # Calculate histogram for each channel (B, G, R)
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    # Normalize histograms
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_r = cv2.normalize(hist_r, hist_r).flatten()

    # Combine histograms into one feature vector (optional)
    hist_combined = np.concatenate([hist_b, hist_g, hist_r])

    histograms[label].append(hist_combined)

print(f"Collected {len(histograms['martensite'])} martensite histograms")
print(f"Collected {len(histograms['bainite'])} bainite histograms")

# Example: Plot average histogram of martensite blue channel
if histograms['martensite']:
    avg_hist_b = np.mean([h[:256] for h in histograms['martensite']], axis=0)
    plt.plot(avg_hist_b, color='blue')
    plt.title('Average Martensite Blue Channel Histogram')
    plt.show()

# Return the histograms dictionary if needed
# You can use it for further analysis or ML tasks
