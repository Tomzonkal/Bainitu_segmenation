import mlflow 
import logging


from mlflow import log_metric, log_param, log_artifacts, log_params
class MLFlowLogger:
    """
    A class to represent an MLFlow Logger.
    
    This class is a placeholder for the actual implementation of an MLFlow logger.
    It should contain methods and properties relevant to logging experiments using MLFlow.
    """
    
    def __init__(self, experiment_name, tracking_uri=None, use_local_backend=True):
        """
        Initializes the MLFlowLogger instance.
        
        :param experiment_name: Name of the experiment
        :param tracking_uri: MLflow tracking URI (optional)
        :param use_local_backend: If True, uses local file backend instead of remote server
        """
        self.experiment_name = experiment_name
        self.iteration = 0
        

        local_uri = f"http://127.0.0.1:5000"
        mlflow.set_tracking_uri(local_uri)
        logging.info(f"Using local MLflow backend: {local_uri}")
        try:
            mlflow.set_experiment(experiment_name)
            logging.info(f"Successfully set experiment: {experiment_name}")
        except Exception as e:
            logging.error(f"Failed to set experiment '{experiment_name}': {e}")
            # Fallback to default experiment
            mlflow.set_experiment("Default")
            logging.info("Falling back to 'Default' experiment")
    def log_dict(self, dict, artifact_file=None):
        """
        Log a dictionary to MLFlow.
        """
        try:
            mlflow.log_dict(dict, artifact_file=artifact_file)
        except Exception as e:
            logging.error(f"Failed to log dictionary: {e}")
        
    def log_params(self, dict):
        """
        Log a parameter to MLFlow.
        
        :param dict: Dictionary of parameters.
        """
        try:
            mlflow.log_params(dict)
        except Exception as e:
            logging.error(f"Failed to log parameter {dict}: {e}")
        
    def log_metric(self, key, value):
        """
        Log a metric to MLFlow.
        
        :param key: Metric name.
        :param value: Metric value.
        :param step: Step number (optional).
        """
        try:
            log_metric(key, value)
        except Exception as e:
            logging.error(f"Failed to log metric {key}: {e}")
        
    def log_artifact(self, local_path, artifact_path=None):
        """
        Log an artifact to MLFlow.
        
        :param local_path: Path to the local file or directory to log as an artifact.
        :param artifact_path: Destination path within the artifact repository (optional).
        """
        try:
            log_artifacts(local_path, artifact_path=artifact_path)
        except Exception as e:
            logging.error(f"Failed to log artifact {local_path}: {e}")
            
    def start_run(self, run_name):
        """
        Start a new MLFlow run.
        """
        try:
            mlflow.start_run(run_name=run_name)
            logging.info(f"Started MLflow run: {run_name}")
        except Exception as e:
            logging.error(f"Failed to start run {run_name}: {e}")
            
    def end_run(self):
        """
        End the current MLFlow run and log the metrics and parameters.
        """
        try:
            mlflow.end_run()
            logging.info("Ended MLflow run")
        except Exception as e:
            logging.error(f"Failed to end run: {e}")






