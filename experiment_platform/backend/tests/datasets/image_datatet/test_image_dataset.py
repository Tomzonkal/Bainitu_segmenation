import pytest
import tempfile
import os
import shutil
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Dodaj ścieżkę do modułów projektu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment_platform_backend.datasets.image_datatet.image_dataset import ImageDataset


class TestImageDataset:
    """Test cases for ImageDataset class using pytest"""
    
    def test_add_metadata(self):
        """Test _add_metadata method"""
        dataset=ImageDataset(
            image_data_path="tests/data/raw_images",
            image_label_data_path="tests/data/metadata",
            dataset_name="test_dataset"
        )
        
        result_df = dataset.load_meta_data()
        
        assert isinstance(result_df, pd.DataFrame)

    
  

