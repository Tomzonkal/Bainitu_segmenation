

class BaseModel:
    """
    Base class for all models in the experiment platform.
    This class provides common functionality that can be shared across different model implementations.
    """

    def __init__(self, input_dataset, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,n_splits=5, shuffle=True, random_state=42):
        self.input_dataset = input_dataset
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _create_histograms(self, segment_dir, histogram_bins):
        """
        Create histograms from the segments in the specified directory.
        :param segment_dir: Directory containing segment images.
        :param histogram_bins: Number of bins for the histogram.
        :return: List of histograms and corresponding labels.
        """
        histograms = []
        labels = []
        
    def prepare_X(self):
        """
        Prepare the feature matrix X from the histograms.
        :param histograms: List of histograms.
        :return: Numpy array of features.
        """
        pass
    def prepare_y(self):
        """
        Prepare the target vector y from the labels.
        :param labels: List of labels.
        :return: Numpy array of targets.
        """
        pass

    def create_segments(self):
        """
        Abstract method to be implemented by subclasses to create segments from the input dataset.
        """
        raise NotImplementedError("Subclasses should implement this method.")