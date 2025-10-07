import re
import uuid
from models import HistogramBaseModel
from optimisers import Optimiser
from preprocessors import SuperpixelSegmentsCreator
from datasets import SegmentDataset
import config
import json
import hashlib
import optuna
import mlflow.sklearn

class OptunaOptimiser(Optimiser):
    """
    A class to represent an Optuna Optimiser.
    
    This class is a placeholder for the actual implementation of an Optuna-based optimiser.
    It should contain methods and properties relevant to the optimisation process using Optuna.
    """
    
    def _produce_model_params(self, trial):
        params = {}
        for param_name in self.model_hyperparameters.keys():
            param=self.model_hyperparameters[param_name]
            if param['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param['min'], param['max'])
            elif param['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param['min'], param['max'])
            elif param['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param['type']}")
        return params
    
    def _produce_slic_params(self, trial):
        params = {}
        for param_name in self.slic_parameters.keys():
            param=self.slic_parameters[param_name]
            if param['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param['min'], param['max'])
            elif param['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param['min'], param['max'])
            elif param['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param['type']}")
        return params
        

    def __init__(self, src_segment_dataset: SegmentDataset, n_trials=100,model_hyperparameters=None,slic_parameters=None,maximize=True,metric_name=None,experiment_name="DefaultExperiment"):
        """
        Initializes the OptunaOptimiser instance.
        """
        self.src_segment_dataset = src_segment_dataset
        self.n_trials = n_trials
        self.model_hyperparameters = model_hyperparameters or []
        self.slic_parameters = slic_parameters or []
        self.slic_output_dataset = None
        self.post_slic_stats = {}
        self.maximize = maximize
        self.metric_name = metric_name
        self.experiment_name = experiment_name+str(uuid.uuid4())
        super().__init__(self.experiment_name)
        if not self.model_hyperparameters:
            raise ValueError("Hyperparameters must be provided for optimisation.")
        if not self.metric_name:
            raise ValueError("Metric name must be provided for optimisation.")
        
    def prepare_slic_segments(self, slic_parameters):
        # Convert params → stable JSON string
        params_str = json.dumps(slic_parameters, sort_keys=True)
        # Hash for short unique name
        hash_name = hashlib.sha1(params_str.encode()).hexdigest()[:12]

        dataset_name = f"slic_{hash_name}"
        slic_out_dataset = SegmentDataset(dataset_name=dataset_name,
                                  image_data_path=config.SUPERPIXEL_BAINITE_SEGMENTS_DATASET_PATH,
                                  image_label_data_path=config.SUPERPIXEL_BAINITE_SEGMENTS_LABELS_DATASET_PATH)
        self.segment_creator = SuperpixelSegmentsCreator(input_dataset=self.src_segment_dataset)
        self.segment_creator.create_segments(slic_parameters, slic_out_dataset) 
        self.post_slic_stats = self.segment_creator.get_post_slic_statistics() 
        self.slic_output_dataset = slic_out_dataset
    
    def prepare_histogram_base_model(self, model_hyperparameters):
        self.model = HistogramBaseModel(input_dataset=self.slic_output_dataset, **model_hyperparameters)

    def log_metrics(self, metrics):
        """
        Log model metrics for the experiment.
        """
        if type(self.model)is HistogramBaseModel:
            metric_list = ["f1","accuracy","precision","recall"]
            for metric in metrics.keys():
                for metric_name in metric_list:
                    self.log_metric(f"{metric}_{metric_name}",metrics[metric][metric_name]) 
        else:
            raise ValueError("Model must be a HistogramBaseModel.")

    def objective(self, trial):
        """
        Objective function for Optuna optimisation.
        
        This function should define the objective to be maximised or minimised during the optimisation process.
        It typically includes hyperparameter tuning and model evaluation.
        """
        model_params=self._produce_model_params(trial)
        slic_params=self._produce_slic_params(trial)

        merged_params = {**model_params, **slic_params}
        self.start_run(f"OptunaOptimiser {merged_params}")
        self.log_params(merged_params)
        self.iteration += 1

        self.prepare_slic_segments(slic_params)
        self.prepare_histogram_base_model(model_params)

        X=self.model.prepare_X()
        y=self.model.prepare_y()

        #try:
        metric,_,_=self.model.train(X, y)
        #except ValueError: # TODO: what to do? think about it
        valid_segments = self.post_slic_stats["valid_segment_counter"]
        martensitic_segment_counter = self.post_slic_stats["martensitic_segment_counter"]
        bainitic_segment_counter = self.post_slic_stats["bainitic_segment_counter"]
        self.log_params(self.post_slic_stats)
        # if valid_segments < 20:
        #     print(f"Valid segment counter is {valid_segments} Skipping trial.")
        #     self.end_run()
        #     return float("-inf") if self.maximize else float("inf")  # Force Optuna to discard this
        
        # if martensitic_segment_counter < 10:
        #     print(f"martensitic_segment_counter segment counter is {martensitic_segment_counter} Skipping trial.")
        #     self.end_run()
        #     return float("-inf") if self.maximize else float("inf")  # Force Optuna to discard this
    
        # if bainitic_segment_counter < 10:
        #     print(f"bainitic_segment_counter segment counter is {bainitic_segment_counter} Skipping trial.")
        #     self.end_run()
        #     return float("-inf") if self.maximize else float("inf")  # Force Optuna to discard this  
    
        self.log_dict(metric, artifact_file=f"metric_{self.iteration}.json")
        # self.log_model_and_metrics(self.model, metric)
        self.log_metrics(metric)
        mlflow.sklearn.log_model(self.model.get_underlying_model()) # TODO: analyze if its possible to maintain consistency with mlflow logger class
        self.end_run()
        return metric["avg_metric"][self.metric_name]


    def optimise(self):
        study = optuna.create_study(direction="maximize" if self.maximize else "minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        print("Best hyperparameters:", study.best_params)
        print("Best accuracy:", study.best_value)
        return study.best_params, study.best_value