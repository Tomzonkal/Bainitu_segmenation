import json
import cv2
import numpy as np
import os
import glob

# === CONFIGURATION ===
metadata_dir = 'metadata'
images_dir = 'raw_images'
output_dir = 'output_segments'

# === SETUP ===
os.makedirs(output_dir, exist_ok=True)

# Find all JSON files in metadata_dir
json_files = glob.glob(os.path.join(metadata_dir, '*.json'))

for json_path in json_files:
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    image_path = os.path.join(images_dir, base_name + '.tif')

    print(f"Processing JSON: {json_path}")
    print(f"Using image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Image not found for {base_name}, skipping...")
        continue

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    label_counts = {}

    # Iterate through shapes in JSON
    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # Extract region using mask
        result = cv2.bitwise_and(image, image, mask=mask)

        # Crop the bounding box of the polygon
        x, y, w, h = cv2.boundingRect(points)
        cropped = result[y:y+h, x:x+w]

        # Naming and saving — prefix filename with base_name to avoid overwriting
        label_counts[label] = label_counts.get(label, 0) + 1
        segment_filename = f"{base_name}_{label}_{label_counts[label]}.png"
        segment_path = os.path.join(output_dir, segment_filename)
        cv2.imwrite(segment_path, cropped)

        print(f"  Saved segment: {segment_path}")

print("Done processing all files.")
