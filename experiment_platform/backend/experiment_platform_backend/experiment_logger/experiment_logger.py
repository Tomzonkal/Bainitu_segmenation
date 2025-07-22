from abc import ABC
from enum import Enum


class LoggingMode(Enum):
    """
    Enum for logging modes.
    """
    MLFLOW = "MLFLOW"
    LOCAL = "LOCAL"


class ExperimentLogger(ABC):
    """
    """
    def log_metrics(self, metrics):
        """
        Log metrics for the experiment.
        
        :param metrics: Dictionary of metrics to log.
        """
        raise NotImplementedError("Subclasses should implement this method")
    def log_params(self, params):
        """
        Log parameters for the experiment.
        
        :param params: Dictionary of parameters to log.
        """
        raise NotImplementedError("Subclasses should implement this method")
    def log_image  (self, image, name):
        """
        Log an image for the experiment.
        
        :param image: Image to log.
        :param name: Name of the image.
        """
        raise NotImplementedError("Subclasses should implement this method")