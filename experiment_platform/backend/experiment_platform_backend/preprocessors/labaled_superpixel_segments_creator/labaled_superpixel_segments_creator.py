import json
import cv2
import numpy as np
import os
from skimage.util import img_as_float
from skimage.segmentation import slic
from datasets.datasets import Dataset
from datasets.generated_dataset import GeneratedDataset

class SuperpixelSegmentsCreator:
    def __init__(self, input_dataset: Dataset, num_superpixels=None, pixels_per_superpixel=50000, compactness=0.1, min_cluster_size=500,sigma=1, start_label=1, channel_axis=None, min_mean_intensity=10):
        self.input_dataset= input_dataset
        self.num_superpixels= num_superpixels if num_superpixels is not None else "AUTO"
        self.pixels_per_superpixel = pixels_per_superpixel
        self.compactness = compactness
        self.min_cluster_size = min_cluster_size
        self.sigma=sigma
        self.start_label = start_label
        self.channel_axis = channel_axis
        self.min_mean_intensity = min_mean_intensity # minimal average pixel value (0-255 scal
        self._skipped_segment_counter = None
        self._valid_segment_counter = None
        self._bainitic_segment_counter = None
        self._martensitic_segment_counter = None
        self.black_threshold = 10         # pixel values < 10 are black
        self.max_black_ratio = 0.9        # skip if > 90% pixels are black

    def _image_preprocessing(self, image, slic_parameters):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert to float for slic (values between 0 and 1) TODO ???
        img_float = img_as_float(gray)
        if self.num_superpixels == "AUTO":
            height, width = img_float.shape[:2]
            num_superpixels = max(2, int((height * width) / slic_parameters["pixels_per_superpixel"]))
        else:
            num_superpixels = self.num_superpixels
        return img_float, num_superpixels


    def _prepare_segment(self, mask, gray):
        # Apply mask to grayscale uint8 image
        masked_img = cv2.bitwise_and(gray, gray, mask=mask)
        ys, xs = np.where(mask == 255)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cropped = masked_img[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        cropped[cropped_mask == 255]
        # if len(xs) == 0 or len(ys) == 0:
        #     return None
        
        if np.count_nonzero(masked_img) == 0:
            print("Segment skipped: fully black",end=" - ")
            return None
        
        # Skip small clusters
        if len(xs) < self.min_cluster_size:
            print(f"Segment skipped: size {len(xs)} < min_cluster_size {self.min_cluster_size}", end=" - ")
            return None

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # # Skip clusters that are mostly black background
        # avg_pixel_intensity = np.mean(cropped)
        # if avg_pixel_intensity < 10:
        #     print(f"Segment mostly black: avg_pixel_intensity: {avg_pixel_intensity}", end=" - ")
        #     return None

        return cropped
    
    def _create_super_pixel_segments(self,label,slic_parameters,path,base_name, output_dataset):
        """
        Create a mask from the given points.
        :param image: The image to create a mask for.
        :param points: Points defining the polygon for the mask.
        :return: Mask as a numpy array.
        """
        base_name_clean = base_name.split('_')[0]
        self.min_cluster_size=slic_parameters["min_cluster_size"]

        image = self._load_image(path)
        img_float, num_superpixels = self._image_preprocessing(image, slic_parameters)
        segments = slic(
            img_float,
            n_segments=num_superpixels,
            compactness=slic_parameters["compactness"],
            sigma=slic_parameters["sigma"],
            start_label=self.start_label,
            channel_axis=self.channel_axis
        )

        local_skipped_segment_counter = 0
        local_valid_segment_counter = 0
        local_bainitic_segment_counter = 0
        local_martensitic_segment_counter = 0
        for segment_iterator, sp_label in enumerate(np.unique(segments)):
            mask = (segments == sp_label).astype(np.uint8) * 255
            segment = self._prepare_segment(mask, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            if segment is not None:
                if label == "bainite":
                    local_bainitic_segment_counter += 1
                if label == "martensite":
                    local_martensitic_segment_counter += 1
                self._save_segment_image(base_name_clean, label, segment_iterator, segment, output_dataset)
                self._save_segment_metadata(base_name_clean, label, segment_iterator, segment, output_dataset)
            else:
                local_skipped_segment_counter += 1
                print(f"{path} Skipped. Label: {label}")
        self._skipped_segment_counter = local_skipped_segment_counter
        self._bainitic_segment_counter = local_bainitic_segment_counter
        self._martensitic_segment_counter = local_martensitic_segment_counter
        self._valid_segment_counter = self._bainitic_segment_counter + self._martensitic_segment_counter


    def get_post_slic_statistics(self):
        return {
            "skipped_segment_counter": self._skipped_segment_counter,
            "bainitic_segment_counter": self._bainitic_segment_counter,
            "martensitic_segment_counter": self._martensitic_segment_counter,
            "valid_segment_counter": self._valid_segment_counter,
        }

    def _load_image(self, image_path):
        """
        Load an image from the given path.
        :param image_path: Path to the image file.
        :return: Loaded image as a numpy array.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        return image
    
    def _load_label_metadata(self, json_path):
        """
        Load JSON data from the given path.
        :param json_path: Path to the JSON file.
        :return: Parsed JSON data.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _save_segment_image(self, base_name, label, iterator, segment, output_dataset):
        """
        Save the extracted segment to the specified path.
        :param segment: The segment to save.
        :param output_path: Path where the segment will be saved.
        """
        # segment_filename = f"{base_name}_{label}_superpixel_segment{iterator}.png"
        segment_filename = f"{base_name}_superpixel_segment{iterator}.png"

        outdir = os.path.join(output_dataset.image_data_path, output_dataset.dataset_name)
        os.makedirs(outdir, exist_ok=True)
        segment_path = os.path.join(outdir, segment_filename)
        cv2.imwrite(segment_path, segment)
    
    def _save_segment_metadata(self, base_name, label, iterator, segment, output_dataset):
        """
        Save the segment metadata to a JSON file.
        :param base_name: Base name for the segment.
        :param label: Label of the segment.
        :param iterator: Iterator for unique naming.
        :param segment: The segment data.
        """
        outdir = os.path.join(output_dataset.image_label_data_path, output_dataset.dataset_name)
        os.makedirs(outdir, exist_ok=True)

        metadata = {
            "base_name": base_name,
            "label": label,
            "iterator": iterator,
            "segment_shape": segment.shape
        }

        metadata_filename = f"{base_name}_superpixel_segment{iterator}.json"
        metadata_path = os.path.join(outdir, metadata_filename)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        

    def create_segments(self, slic_parameters, output_dataset: GeneratedDataset):
        metadata = self.input_dataset.load_meta_data()
        for _, row in metadata.iterrows():
            image_path = row['image_path']
            json_path = row['json_path']
            if json_path is None:
                print(f"Warning: No JSON path for {image_path}, skipping...")
                continue
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            image_label_metadata = self._load_label_metadata(json_path)
            self._create_super_pixel_segments(
                image_label_metadata['label'],
                slic_parameters,
                image_path,
                base_name,
                output_dataset
            )