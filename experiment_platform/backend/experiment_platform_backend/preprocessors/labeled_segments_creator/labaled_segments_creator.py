import json
import cv2
import numpy as np
import os
import glob

from experiment_platform_backend.datasets.datasets import Dataset
class LabeledSegmentsCreator:
    def __init__(self, input_dataset: str, output_dataset:Dataset):
        self.input_dataset= input_dataset
        self.output_dataset = output_dataset
        
    def _create_segment(self, image, points):
        """
        Create a mask from the given points.
        :param image: The image to create a mask for.
        :param points: Points defining the polygon for the mask.
        :return: Mask as a numpy array.
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        # Extract region using mask
        result = cv2.bitwise_and(image, image, mask=mask)

        # Crop the bounding box of the polygon
        x, y, w, h = cv2.boundingRect(points)
        cropped = result[y:y+h, x:x+w]
        return cropped

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
    
    def _save_segment_image(self, base_name, label,iterator,segment):
        """
        Save the extracted segment to the specified path.
        :param segment: The segment to save.
        :param output_path: Path where the segment will be saved.
        """                
        segment_filename = f"{base_name}_{label}_{iterator}.png"
        segment_path = os.path.join(self.output_dataset.image_data_path, segment_filename)
        cv2.imwrite(segment_path, segment)
    
    def _save_segment_metadata(self, base_name, label, iterator, segment):
        """
        Save the segment metadata to a JSON file.
        :param base_name: Base name for the segment.
        :param label: Label of the segment.
        :param iterator: Iterator for unique naming.
        :param segment: The segment data.
        """
        metadata = {
            'base_name': base_name,
            'label': label,
            'iterator': iterator,
            'segment_shape': segment.shape
        }
        metadata_filename = f"{base_name}_{label}_{iterator}.json"
        metadata_path = os.path.join(self.output_dataset.image_label_data_path, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        

    def create_segments(self):
        metadta = self.input_dataset.load_meta_data()
        index=0
        for index, row in metadta.iterrows():
            image_path = row['image_path']
            json_path = row['json_path']
            
            if json_path is None:
                print(f"Warning: No JSON path for {image_path}, skipping...")
                continue
            
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            image=self._load_image(image_path)
            image_label_metadta = self._load_label_metadata(json_path)
            # Iterate through shapes in JSON
            for label_metadata in image_label_metadta['shapes']:
                index=index+1
                label = label_metadata['label']
                segment= self._create_segment(image, np.array(label_metadata['points'], dtype=np.int32))
                self._save_segment_image(base_name, label, index, segment)
                self._save_segment_metadata(base_name, label, index, segment)
        print(f"Created {index} segments from {len(metadta)} images.")