from abc import ABC, abstractmethod



class Preprocessor(ABC):    
    """
    Abstract base class for preprocessors.
    """
    
    def run(self):
        """
        Run the preprocessor on the input data.
        
        :param data: Input data to preprocess.
        :return: Preprocessed data.
        """
        data = self.read_data(data)
        if not self.validate(data):
            raise ValueError("Invalid input data")
        
        processed_data = self.process(data)
        self.write_data(processed_data, "processed_data.json")
        return processed_data
    
    def validate(self, data):
        """
        Validate the input data.
        
        :param data: Input data to validate.
        :return: True if valid, False otherwise.
        """
        raise NotImplementedError("Subclasses should implement this method")
    
    def process(self, data):
        """
        Process the input data.
        
        :param data: Input data to process.
        :return: Processed data.
        """
        raise NotImplementedError("Subclasses should implement this method")
    def read_data(self, path):
        """
        Read data from a file.
        
        :param path: Path to the file.
        :return: Data read from the file.
        """
        raise NotImplementedError("Subclasses should implement this method")
    def write_data(self, data, path):
        """ 
        Write data to a file.
        :param data: Data to write.
        :param path: Path to the file.
        """
        raise NotImplementedError("Subclasses should implement this method")