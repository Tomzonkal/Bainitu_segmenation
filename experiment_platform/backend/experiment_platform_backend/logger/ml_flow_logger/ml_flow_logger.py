import mlflow 


from mlflow import log_metric, log_param, log_artifacts
class MLFlowLogger:
    """
    A class to represent an MLFlow Logger.
    
    This class is a placeholder for the actual implementation of an MLFlow logger.
    It should contain methods and properties relevant to logging experiments using MLFlow.
    """
    
    def __init__(self, experiment_name, run_name=None):
        """
        Initializes the MLFlowLogger instance.
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        
    def log_param(self, key, value):
        """
        Log a parameter to MLFlow.
        
        :param key: Parameter name.
        :param value: Parameter value.
        """
        log_param(key, value)
        
    def log_metric(self, key, value, step=None):
        """
        Log a metric to MLFlow.
        
        :param key: Metric name.
        :param value: Metric value.
        :param step: Step number (optional).
        """
        log_metric(key, value, step=step)
        
    def log_artifact(self, local_path, artifact_path=None):
        """
        Log an artifact to MLFlow.
        
        :param local_path: Path to the local file or directory to log as an artifact.
        :param artifact_path: Destination path within the artifact repository (optional).
        """
        log_artifacts(local_path, artifact_path=artifact_path)
        
    def end_run(self):
        """
        End the current MLFlow run.
        """
        mlflow.end_run()





