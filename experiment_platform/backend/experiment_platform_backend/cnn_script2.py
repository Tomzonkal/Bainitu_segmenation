from datasets import ImageDataset, SegmentDataset
from preprocessors import LabeledSegmentsCreator,SuperpixelSegmentsCreator
from models import HistogramBaseModel
import config
import numpy as np
import mlflow
# Initialize the image dataset with the specified paths
image_dataset=ImageDataset(dataset_name="raw_bainite_dataset",
                 image_data_path=config.RAW_BAINITE_IMAGE_DATASET_PATH,
                 image_label_data_path=config.RAW_BAINITE_LABEL_DATASET_PATH,
    )
raw_images_df=image_dataset.load_meta_data()
raw_images_df = raw_images_df[raw_images_df['json_path'].notnull()]

# Initialize the segment dataset
segment_dataset = SegmentDataset(dataset_name="raw_bainite_segments",
                                  image_data_path=config.RAW_BAINITE_SEGMENTS_DATASET_PATH,
                                  image_label_data_path=config.RAW_BAINITE_SEGMENTS_LABELS_DATASET_PATH)
# Create the labeled segments
labeled_segments_creator = LabeledSegmentsCreator(input_dataset=image_dataset, output_dataset=segment_dataset)
labeled_segments_creator.create_segments()
segment_dataset.load_meta_data()

from optimisers.optuna_optimiser.optuna_optimiser import OptunaOptimiser


model_parameters = {
    'lr': {'type': 'float', 'min': 2e-2, 'max': 8e-2},
    'epochs': {'type': 'int', 'min': 20, 'max': 60},
}

slic_parameters = {
        "pixels_per_superpixel" : {'type' : 'int', 'min' : 14000, 'max' : 20000},
        "compactness" : {'type' : 'float', 'min' : 0.65, 'max' : 0.75}, # consider 0.5
        "sigma" : {'type' : 'float', 'min' : 1, 'max' : 1.4}
}

optuna_optimiser= OptunaOptimiser(src_segment_dataset=segment_dataset, model_name="efficientnet_b0", segmentation="slic", n_trials=50,
                                   model_hyperparameters=model_parameters, segmentation_parameters = slic_parameters, maximize=True, metric_name='f1', experiment_name="efficientnet_b0_scratch")
optuna_optimiser.optimise()