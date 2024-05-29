# Data Loading Module

import torch
from .preprocess import load_and_preprocess_data

class DataLoader:
    """
    Data Loading Class.

    Args:
        data_path (str): Path to the CSV file.

    Attributes:
        inputs_train (numpy.ndarray): Preprocessed input data for training.
        targets_train (numpy.ndarray): Preprocessed target data for training.
        inputs_test (numpy.ndarray): Preprocessed input data for testing.
        targets_test (numpy.ndarray): Preprocessed target data for testing.
        scaler (StandardScaler): Scaler used for scaling the data.
    """
    def __init__(self, data_path):
        self.inputs_train, self.targets_train, self.inputs_test, self.targets_test, self.scaler = load_and_preprocess_data(data_path)

    def get_train_data(self):
        """
        Returns the training data as a PyTorch tensor.

        Returns:
            A tuple containing the input data and the target data as PyTorch tensors.
        """
        return torch.tensor(self.inputs_train, dtype=torch.float32), torch.tensor(self.targets_train, dtype=torch.long)

    def get_test_data(self):
        """
        Returns the test data as a PyTorch tensor.

        Returns:
            A tuple containing the input data and the target data as PyTorch tensors.
        """
        return torch.tensor(self.inputs_test, dtype=torch.float32), torch.tensor(self.targets_test, dtype=torch.long)

    def inverse_transform(self, inputs):
        """
        Inverse transforms the input data using the scaler.

        Args:
            inputs (numpy.ndarray): Input data to be inverse transformed.

        Returns:
            The inverse transformed input data.
        """
        return self.scaler.inverse_transform(inputs)
