import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from experiment_platform.backend.src.get_histograms import get_histograms
from experiment_platform.backend.src.create_superpixels import create_superpixels
from experiment_platform.backend.src.opt import objective, callback
import optuna

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("RF Optuna Experiment whit histogram bins")

with mlflow.start_run(run_name="optuna_rf_cv_aggregated_metrics") as parent_run:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, callbacks=[callback])
    best_model=study.user_attrs["best_booster"]

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_f1_score", study.best_value)

    print("Best trial parameters:", study.best_trial.params)
    print(f"Best F1 score: {study.best_value:.4f}")
    best_trial_number = study.best_trial.number

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y)

# clf = RandomForestClassifier(**model_params)
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(classification_report(y_test, y_pred, target_names=['Bainite', 'Martensite']))



# # Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# # Create a new MLflow Experiment
# mlflow.set_experiment("RF Initial Experiment")

# # Start an MLflow run
# with mlflow.start_run():
#     print("MLFlow started")
#     mlflow.log_params(segmentation_params)
#     mlflow.log_param("segment_count", segment_count)
#     mlflow.log_param("sample_images_count", sample_images_count)
#     mlflow.log_param("segments_per_image", int(segment_count / sample_images_count))
#     mlflow.log_params(model_params)

#     mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

#     signature = infer_signature(X_train, clf.predict(X_train))

#     model_info = mlflow.sklearn.log_model(
#         sk_model=clf,
#         name="rf_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="rf",
#     )
#     print("model registered")

#     # Set a tag that we can use to remind ourselves what this model was for
#     mlflow.set_logged_model_tags(
#         model_info.model_id, {"Training Info": "Basic RF model for bainite/martensite"}
#     )
