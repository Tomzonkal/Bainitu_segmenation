import re
from datasets.datasets import Dataset
import json
import os
import glob
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self,image_data_path:str, image_label_data_path:str, dataset_name:str):
        """
        Initialize the ImageDataset class.
        """
        super().__init__(image_data_path=image_data_path, image_label_data_path=image_label_data_path,dataset_name = dataset_name)
        

    def _add_label_infomration(self, df:pd.DataFrame):
        for index, row in df.iterrows():
            json_path = row['json_path']
            if json_path is not None:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    martensite_count = sum(1 for shape in data.get('shapes', []) if shape.get('label') == 'martensite')
                    bainite_count = sum(1 for shape in data.get('shapes', []) if shape.get('label') == 'bainite')
                    df.at[index, 'martensite_count'] = martensite_count
                    df.at[index, 'bainite_count'] = bainite_count

            else:
                df.at[index, 'martensite_count'] = None
                df.at[index, 'bainite_count'] = None
        return df
        
    
    def _add_json_paths(self, images_paths,df):
        json_paths = glob.glob(os.path.join(self.image_label_data_path, '*.json'))
        df['json_path'] = None
        for image_path in images_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(self.image_label_data_path, base_name + '.json')
            df.loc[df['image_path'] == image_path, 'json_path'] = json_path if json_path in json_paths else None
        return df
            
    
    def _add_metadata(self, images_paths:list):
        """
        Create a DataFrame with metadata from the dataset.
        
        :return: DataFrame containing image paths and labels.
        """
        df = pd.DataFrame(images_paths, columns=['image_path'])
        df = self._add_json_paths(images_paths,df)
        df=self._add_label_infomration(df)
        return df
        
        
    def load_meta_data(self)-> pd.DataFrame:
        """
        Load metadata from a specified path.
        
        :param labeled_only: If True, return only images with corresponding JSON labels.
        :return: DataFrame containing image paths and metadata.
        """
        # Find all TIF files in image_data_path
        images_paths = glob.glob(os.path.join(self.image_data_path, '*.tif'))
        df = self._add_metadata(images_paths)
        return df