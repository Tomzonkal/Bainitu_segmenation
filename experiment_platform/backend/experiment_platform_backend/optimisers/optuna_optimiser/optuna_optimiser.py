import re
import uuid
from models import HistogramBaseModel, ResNet50Model, EfficientNetB0, CustomGrayCNN, Custom1LayerGrayCNN
from optimisers import Optimiser
from preprocessors import SuperpixelSegmentsCreator, FelzenszwalbSegmentsCreator, WatershedSegmentsCreator, GridSegmentsCreator
from datasets import SegmentDataset
import config
import json
import hashlib
import optuna
import mlflow.sklearn
import mlflow.pytorch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
    
    def _produce_segmentation_params(self, trial):
        params = {}
        for param_name in self.segmentation_parameters.keys():
            param=self.segmentation_parameters[param_name]
            if param['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param['min'], param['max'])
            elif param['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param['min'], param['max'])
            elif param['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param['type']}")
        return params
        

    def __init__(self, src_segment_dataset: SegmentDataset, model_name, segmentation, n_trials=100,model_hyperparameters=None,segmentation_parameters=None,maximize=True,metric_name=None,experiment_name="DefaultExperiment"):
        """
        Initializes the OptunaOptimiser instance.
        """
        self.src_segment_dataset = src_segment_dataset
        self.model_name=model_name
        self.model=None
        self.segmentation = segmentation
        self.n_trials = n_trials
        self.model_hyperparameters = model_hyperparameters or []
        self.segmentation_parameters = segmentation_parameters or []
        self.segmentation_output_dataset = None
        self.post_segmentation_stats = {}
        self.maximize = maximize
        self.metric_name = metric_name
        self.experiment_name = experiment_name+str(uuid.uuid4())
        super().__init__(self.experiment_name)
        if not self.model_hyperparameters:
            raise ValueError("Hyperparameters must be provided for optimisation.")
        if not self.metric_name:
            raise ValueError("Metric name must be provided for optimisation.")
        
    def prepare_segments(self, segmentation_parameters):
        # Convert params → stable JSON string
        params_str = json.dumps(segmentation_parameters, sort_keys=True)
        # Hash for short unique name
        hash_name = hashlib.sha1(params_str.encode()).hexdigest()[:12]
        self.log_dict({"Dataset name" : hash_name}, artifact_file=f"dataset_{hash_name}.json")

        dataset_name = f"{self.segmentation}_{hash_name}"
        segmentation_out_dataset = SegmentDataset(dataset_name=dataset_name,
                                  image_data_path=config.SUPERPIXEL_BAINITE_SEGMENTS_DATASET_PATH,
                                  image_label_data_path=config.SUPERPIXEL_BAINITE_SEGMENTS_LABELS_DATASET_PATH)
        if self.segmentation == "slic":
            self.segment_creator = SuperpixelSegmentsCreator(input_dataset=self.src_segment_dataset)
        elif self.segmentation == "felzenszwalb":
            self.segment_creator = FelzenszwalbSegmentsCreator(input_dataset=self.src_segment_dataset)
        elif self.segmentation == "watershed":
            self.segment_creator = WatershedSegmentsCreator(input_dataset=self.src_segment_dataset)
        elif self.segmentation == "grid":
            self.segment_creator = GridSegmentsCreator(input_dataset=self.src_segment_dataset)
            print("Debug: segment creator set to grid")
        else:
            raise ValueError("Segmentation must be slic, felzenszwalb, watershed or grid")
        self.segment_creator.create_segments(segmentation_parameters, segmentation_out_dataset) 
        print("Debug: segment creation done")
        self.post_segmentation_stats = self.segment_creator.get_post_segmentation_statistics() 
        self.segmentation_output_dataset = segmentation_out_dataset
    
    def prepare_histogram_base_model(self, model_hyperparameters):
        self.model = HistogramBaseModel(input_dataset=self.segmentation_output_dataset, **model_hyperparameters)
    
    def prepare_resnet50_model(self, model_hyperparameters):
        self.model = ResNet50Model(input_dataset=self.segmentation_output_dataset, **model_hyperparameters)
        
    def prepare_efficientnet_b0_model(self, model_hyperparameters):
        self.model = EfficientNetB0(input_dataset=self.segmentation_output_dataset, **model_hyperparameters)
        
    def prepare_custom_cnn_model(self, model_hyperparameters):
        self.model = CustomGrayCNN(input_dataset=self.segmentation_output_dataset, **model_hyperparameters)

    def prepare_custom_1_layer_cnn_model(self, model_hyperparameters):
        self.model = Custom1LayerGrayCNN(input_dataset=self.segmentation_output_dataset, **model_hyperparameters)

    def log_metrics(self, metrics):
        """
        Log model metrics for the experiment.
        """
        # if type(self.model)is HistogramBaseModel:
        metric_list = ["f1","accuracy","precision","recall"]
        for metric in metrics.keys():
            for metric_name in metric_list:
                self.log_metric(f"{metric}_{metric_name}",metrics[metric][metric_name]) 
        # else:
        #     raise ValueError("Model must be a HistogramBaseModel.")

    def objective(self, trial):
        """
        Objective function for Optuna optimisation.
        
        This function should define the objective to be maximised or minimised during the optimisation process.
        It typically includes hyperparameter tuning and model evaluation.
        """
        model_params=self._produce_model_params(trial)
        segmentation_params=self._produce_segmentation_params(trial)

        merged_params = {**model_params, **segmentation_params}
        self.start_run(f"OptunaOptimiser {merged_params}")
        self.log_params(merged_params)
        self.iteration += 1

        self.prepare_segments(segmentation_params)
        if self.model_name == "HistogramBaseModel":
            self.prepare_histogram_base_model(model_params)
        elif self.model_name == "ResNet50":
            self.prepare_resnet50_model(model_params)
        elif self.model_name == "efficientnet_b0":
            self.prepare_efficientnet_b0_model(model_params)
        elif self.model_name == "custom_cnn":
            self.prepare_custom_cnn_model(model_params)
        elif self.model_name == "custom_1layer_cnn":
            self.prepare_custom_1_layer_cnn_model(model_params)

        X=self.model.prepare_X()
        y=self.model.prepare_y()

        #try:
        metric,cm,_=self.model.train(X, y)
        #except ValueError: # TODO: what to do? think about it
        valid_segments = self.post_segmentation_stats["valid_segment_counter"]
        martensitic_segment_counter = self.post_segmentation_stats["martensitic_segment_counter"]
        bainitic_segment_counter = self.post_segmentation_stats["bainitic_segment_counter"]
        self.log_params(self.post_segmentation_stats)

        disp = ConfusionMatrixDisplay(cm)
        disp.plot()

        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

        self.log_artifact("confusion_matrix.png")
        if valid_segments < 50:
            print(f"Valid segment counter is {valid_segments} Skipping trial.")
            self.end_run()
            return float("-inf") if self.maximize else float("inf")  # Force Optuna to discard this
        
        if martensitic_segment_counter < 20:
            print(f"martensitic_segment_counter segment counter is {martensitic_segment_counter} Skipping trial.")
            self.end_run()
            return float("-inf") if self.maximize else float("inf")  # Force Optuna to discard this
    
        if bainitic_segment_counter < 20:
            print(f"bainitic_segment_counter segment counter is {bainitic_segment_counter} Skipping trial.")
            self.end_run()
            return float("-inf") if self.maximize else float("inf")  # Force Optuna to discard this  
    
        self.log_dict(metric, artifact_file=f"metric_{self.iteration}.json")
        # self.log_model_and_metrics(self.model, metric)
        self.log_metrics(metric)
        if self.model_name == "ResNet50" or self.model_name == "efficientnet_b0":
            mlflow.pytorch.log_model(self.model.get_underlying_model()) # TODO: analyze if its possible to maintain consistency with mlflow logger class
        if self.model_name == "HistogramBaseModel":
            mlflow.sklearn.log_model(self.model.get_underlying_model()) # TODO: analyze if its possible to maintain consistency with mlflow logger class
        self.end_run()
        return metric["avg_metric"][self.metric_name]


    def optimise(self):
        study = optuna.create_study(direction="maximize" if self.maximize else "minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        print("Best hyperparameters:", study.best_params)
        print("Best accuracy:", study.best_value)
        return study.best_params, study.best_value

    def predict_on_test_dataset(self, test_dataset: SegmentDataset, model_path: str, segmentation_params):
        # best_params = self.study.best_params
        # model_params = {
        #     k: best_params[k]
        #     for k in best_params.keys()
        #     if k in self.model_hyperparameters.keys()
        # }
        # segmentation_params = {
        #     k: best_params[k]
        #     for k in best_params.keys()
        #     if k in self.segmentation_parameters.keys()
        # }

        
        self.prepare_segments(segmentation_params)
        
        # Rebuild model architecture wrapper
        if self.model_name == "HistogramBaseModel":
            self.prepare_histogram_base_model(self.model_hyperparameters)
            underlying_model = mlflow.sklearn.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Inject the loaded model weights into the wrapper
        self.model.set_underlying_model(underlying_model)
        
        # Prepare segments for test dataset (same segmentation settings)
        

        # Prepare features and labels
        X_test = self.model.prepare_X()
        y_test = self.model.prepare_y()
        
        print("Len of x:", len(X_test))
        print("Len of y:", len(y_test))
        
        # Generate predictions
        print("🚀 Running inference on test dataset...")
        predictions = self.model.get_underlying_model().predict(X_test)

        # Run prediction and evaluation -- evaluate is for tocrch
        # metrics, predictions, confusion_matrix = self.model.get_underlying_model().evaluate(X_test, y_test)
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)

        # # Optionally compute confusion matrix
        # if hasattr(self.model, "compute_confusion_matrix"):
        #     confusion_matrix = self.model.compute_confusion_matrix(y_test, predictions)
        # else:
        #     confusion_matrix = None

        # # Log and return
        # self.start_run("Test Prediction Run")
        # self.log_dict(metrics, artifact_file="test_metrics.json")
        # self.log_dict(confusion_matrix, artifact_file="confusion_matrix.json")
        # self.log_metrics(metrics)
        # self.end_run()

        print("✅ Test dataset evaluation completed.")