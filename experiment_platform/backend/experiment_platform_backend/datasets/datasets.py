from abc import ABC


class Dataset(ABC):
    """
    Abstract base class for datasets.
    """
    def __init__(self,image_data_path, image_label_data_path,dataset_name):
        """
        Initialize the Dataset class.
        """
        self.image_data_path = image_data_path
        self.image_label_data_path = image_label_data_path
        self.dataset_name = dataset_name
        
    def load_meta_data(self):
        """
        Load metadata from a specified path.
        
        :param labeled_only: If True, return only images with corresponding JSON labels.
        :return: DataFrame containing image paths and metadata.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
