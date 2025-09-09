from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ROOT_DATA_DIR = (BASE_DIR / ".." / ".." / ".." / "data_labeling" / "src").resolve()

RAW_BAINITE_IMAGE_DATASET_PATH = ROOT_DATA_DIR / "raw_images"
RAW_BAINITE_LABEL_DATASET_PATH = ROOT_DATA_DIR / "metadata"
RAW_BAINITE_SEGMENTS_DATASET_PATH = ROOT_DATA_DIR / "results"
RAW_BAINITE_SEGMENTS_LABELS_DATASET_PATH = ROOT_DATA_DIR / "results"
SUPERPIXEL_BAINITE_SEGMENTS_DATASET_PATH = ROOT_DATA_DIR / "results"
SUPERPIXEL_BAINITE_SEGMENTS_LABELS_DATASET_PATH = ROOT_DATA_DIR / "results"

# TODO:
# RAW_BAINITE_IMAGE_DATASET_PATH="../../../data_labeling/src/raw_images"
# RAW_BAINITE_LABEL_DATASET_PATH="../../../data_labeling/src/metadata"
# RAW_BAINITE_SEGMENTS_DATASET_PATH="../../../data_labeling/src/results"
# RAW_BAINITE_SEGMENTS_LABELS_DATASET_PATH="../../../data_labeling/src/results"
# SUPERPIXEL_BAINITE_SEGMENTS_DATASET_PATH="../../../data_labeling/src/results"
# SUPERPIXEL_BAINITE_SEGMENTS_LABELS_DATASET_PATH="../../../data_labeling/src/results"

LOGGING_MODE="MLFLOW" # or LOCAL"