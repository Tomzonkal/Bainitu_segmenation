import cv2
from skimage.segmentation import slic, mark_boundaries, felzenszwalb
from skimage.util import img_as_float
import os
import shutil
import numpy as np
import glob


def create_superpixels(segment_dir, output_dir, pixels_per_superpixel=50000, compactness=0.1, min_cluster_size=500):
    segment_files = glob.glob(os.path.join(segment_dir, '*.png'))
    # clear content od directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    num_superpixels_counter, full_sample_img_counter = 0, 0
    for seg_path in segment_files:
        img_bgr = cv2.imread(seg_path)
        if img_bgr is None:
            print(f"Could not read image: {seg_path}")
            continue
        full_sample_img_number = int(seg_path.split("x-")[1].split("_")[0])
        if full_sample_img_number > full_sample_img_counter:
            full_sample_img_counter = full_sample_img_number

        height, width = img_bgr.shape[:2]

        # Convert to grayscale uint8
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Convert to float for slic (values between 0 and 1)
        img_float = img_as_float(gray)

        num_superpixels = max(2, int((height * width) / pixels_per_superpixel)) # TODO optimize, register

        print(f"{os.path.basename(seg_path)} - Image size: {width}x{height}, SLIC segments: {num_superpixels}")

        # Run SLIC on grayscale (2D array), set channel_axis=None explicitly for grayscale images TODO optimize, register
        segments = slic(img_float, n_segments=num_superpixels, compactness=compactness, sigma=1, start_label=1, channel_axis=None)
        # img_with_boundaries = mark_boundaries(img_float, segments)

        base_filename = os.path.splitext(os.path.basename(seg_path))[0]

        MIN_MEAN_INTENSITY = 10  # minimal average pixel value (0-255 scale) to consider "not background" TODO optimize, register

        for sp_label in np.unique(segments):
            mask = (segments == sp_label).astype(np.uint8) * 255

            # Apply mask to grayscale uint8 image
            masked_img = cv2.bitwise_and(gray, gray, mask=mask)

            ys, xs = np.where(mask == 255)
            if len(xs) == 0 or len(ys) == 0:
                continue

            # Skip small clusters
            if len(xs) < min_cluster_size:
                continue

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cropped = masked_img[y_min:y_max+1, x_min:x_max+1]

            # Skip clusters that are mostly black background
            mean_intensity = np.mean(cropped)
            if mean_intensity < MIN_MEAN_INTENSITY:
                continue

            out_filename = f"{base_filename}_sp{sp_label}.png"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, cropped)
        num_superpixels_counter += num_superpixels
        print(f"Saved {num_superpixels} superpixels from {base_filename}")
    return num_superpixels_counter, full_sample_img_counter
