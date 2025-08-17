import re
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
        

    def __init__(self,model,input_dataset,n_trials=100,hyper_parameters=None,maximize=True,metric_name=None):
        """
        Initializes the OptunaOptimiser instance.
        """
        self.model = model
        self.input_dataset = input_dataset
        self.n_trials = n_trials
        self.hyper_parameters = hyper_parameters or []
        self.maximize = maximize
        self.metric_name = metric_name
        if not self.hyper_parameters:
            raise ValueError("Hyperparameters must be provided for optimisation.")
        if not self.metric_name:
            raise ValueError("Metric name must be provided for optimisation.")

    def objective(self, trial):
        """
        Objective function for Optuna optimisation.
        
        This function should define the objective to be maximised or minimised during the optimisation process.
        It typically includes hyperparameter tuning and model evaluation.
        """
        params=self._produce_params(trial)
        # Set model parameters
        model=self.model(input_dataset=self.input_dataset, **params)
        # Fit the model and evaluate

        X=model.prepare_X()
        y=model.prepare_y()

        metric,_,_=model.train(X, y)
        return metric[self.metric_name]


    def optimise(self):
        study = optuna.create_study(direction="maximize" if self.maximize else "minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        print("Best hyperparameters:", study.best_params)
        print("Best accuracy:", study.best_value)
        return study.best_params, study.best_value