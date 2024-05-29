# Data Preprocessing Module

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(data_path):
    """
    Loads data from a CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        A tuple containing the input data and the target data.
    """
    data = pd.read_csv(data_path)
    inputs = data.iloc[:, :-1].values
    targets = data.iloc[:, -1].values
    return inputs, targets

def preprocess_data(inputs, targets):
    """
    Preprocesses the data by scaling the inputs and splitting the data into training and validation sets.

    Args:
        inputs (numpy.ndarray): Input data.
        targets (numpy.ndarray): Target data.

    Returns:
        A tuple containing the preprocessed inputs, the preprocessed targets, and the test set.
    """
    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)
    return inputs_train, targets_train, inputs_test, targets_test, scaler

def load_and_preprocess_data(data_path):
    """
    Loads and preprocesses the data.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        A tuple containing the preprocessed inputs, the preprocessed targets, and the test set.
    """
    inputs, targets = load_data(data_path)
    inputs_train, targets_train, inputs_test, targets_test, scaler = preprocess_data(inputs, targets)
    return inputs_train, targets_train, inputs_test, targets_test, scaler
