import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow.models import infer_signature
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from create_superpixels import create_superpixels
from get_histograms import get_histograms
from PIL import Image
import joblib

segment_dir = "output_segments"
slic_out_dir = "slic_out"


def plot_confusion_matrix(cm, labels):

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Convert buffer to a PIL Image for MLflow
    img = Image.open(buf)
    return img

def objective(trial):
    histogram_bins = trial.suggest_int("histogram_bins", 1, 256)
    segmentation_params = {
        "pixels_per_superpixel" : trial.suggest_int("pixels_per_superpixel",50, 50000),
        "compactness" : trial.suggest_float("compactness", 0.1, 1),
        "min_cluster_size" : trial.suggest_int("min_cluster_size", 100, 10000),
    }
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]) # TODO ????
    }
        # Step 1: Get data
    histograms, labels = get_histograms(slic_out_dir, histogram_bins)

    # Step 2: Create train/test split
    X = np.array(histograms)
    y = np.array(labels)

    segment_count, sample_images_count = create_superpixels(segment_dir, slic_out_dir, **segmentation_params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true_all = []
    y_pred_all = []

    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Processing {i} fold...")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        clf = RandomForestClassifier(**params, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)

        y_true_all.extend(y_valid)
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Calculate metrics on combined predictions from all folds
    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    labels = np.unique(y)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

    with mlflow.start_run(nested=True):
        mlflow.log_params(segmentation_params)
        mlflow.log_params(params)
        mlflow.log_param("histogram_bins", histogram_bins)
        mlflow.log_param("segment_count", segment_count)
        mlflow.log_param("sample_images_count", sample_images_count)
        mlflow.log_param("segments_per_image", int(segment_count / sample_images_count))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)


        # Log confusion matrix plot as artifact
        buf = plot_confusion_matrix(cm, labels)
        mlflow.log_image(buf, "confusion_matrix.png")
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

        trial.set_user_attr(key="best_booster", value=clf)

        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_logged_model_tags(
            model_info.model_id, {"Training Info": "Basic RF model for bainite/martensite"}
        )


    print("Leaving objective fun")
    return f1  # or your preferred metric


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])