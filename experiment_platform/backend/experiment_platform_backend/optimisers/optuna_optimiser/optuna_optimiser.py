import re
import uuid
from models import HistogramBaseModel
from optimisers import Optimiser
import optuna
class OptunaOptimiser(Optimiser):
    """
    A class to represent an Optuna Optimiser.
    
    This class is a placeholder for the actual implementation of an Optuna-based optimiser.
    It should contain methods and properties relevant to the optimisation process using Optuna.
    """
    
    def _produce_params(self, trial):
        params = {}
        for param_name in self.hyper_parameters.keys():
            param=self.hyper_parameters[param_name]
            if param['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param['min'], param['max'])
            elif param['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param['min'], param['max'])
            elif param['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param['type']}")
        return params
        

    def __init__(self,model,input_dataset,n_trials=100,hyper_parameters=None,maximize=True,metric_name=None,experiment_name="DefaultExperiment"):
        """
        Initializes the OptunaOptimiser instance.
        """
        self.model = model
        self.input_dataset = input_dataset
        self.n_trials = n_trials
        self.hyper_parameters = hyper_parameters or []
        self.maximize = maximize
        self.metric_name = metric_name
        self.experiment_name = experiment_name+str(uuid.uuid4())
        super().__init__(self.experiment_name)
        if not self.hyper_parameters:
            raise ValueError("Hyperparameters must be provided for optimisation.")
        if not self.metric_name:
            raise ValueError("Metric name must be provided for optimisation.")
        
        
    def log_model_metrics(self, model,metrics):
        """
        Log model metrics for the experiment.
        """
        if type(model)is HistogramBaseModel:
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
        params=self._produce_params(trial)
        self.start_run(f"OptunaOptimiser {params}")
        self.log_params(params)
        self.iteration += 1
        # Set model parameters
        model=self.model(input_dataset=self.input_dataset, **params)
        # Fit the model and evaluate

        X=model.prepare_X()
        y=model.prepare_y()

        metric,_,_=model.train(X, y)
        self.log_dict(metric, artifact_file=f"metric_{self.iteration}.json")
        self.log_model_metrics(model,metric)
        self.end_run()
        return metric["avg_metric"][self.metric_name]


    def optimise(self):
        study = optuna.create_study(direction="maximize" if self.maximize else "minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        print("Best hyperparameters:", study.best_params)
        print("Best accuracy:", study.best_value)
        return study.best_params, study.best_value