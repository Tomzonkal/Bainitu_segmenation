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
from get_histograms import get_histograms
from create_superpixels import create_superpixels

segment_dir = "output_segments"
slic_out_dir = "slic_out"

segmentation_params = {

    "pixels_per_superpixel" : 50000,
    "compactness" : 0.5,
    "min_cluster_size" : 500

}
model_params = {

    "n_estimators" : 100,
    "max_depth" : 100
}

# Step 1: Get data
segment_count, sample_images_count = create_superpixels(segment_dir, slic_out_dir, **segmentation_params)
histograms, labels = get_histograms(slic_out_dir)

# Step 2: Create train/test split
X = np.array(histograms)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train Random Forest classifier
clf = RandomForestClassifier(**model_params)
clf.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Bainite', 'Martensite']))



# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("RF Initial Experiment")

# Start an MLflow run
with mlflow.start_run():
    print("MLFlow started")
    # Log the hyperparameters
    mlflow.log_params(segmentation_params)
    mlflow.log_param("segment_count", segment_count)
    mlflow.log_param("sample_images_count", sample_images_count)
    mlflow.log_param("segments_per_image", int(segment_count / sample_images_count))
    mlflow.log_params(model_params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # Infer the model signature
    signature = infer_signature(X_train, clf.predict(X_train))

    # Log the model, which inherits the parameters and metric
    model_info = mlflow.sklearn.log_model(
        sk_model=clf,
        name="rf_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="rf",
    )
    print("model registered")

    # Set a tag that we can use to remind ourselves what this model was for
    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Basic RF model for bainite/martensite"}
    )
