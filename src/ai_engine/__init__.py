# Import necessary modules and initialize the AI engine
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Set up the AI engine configuration
AI_ENGINE_CONFIG = {
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'seed': 42,
    'batch_size': 128,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_fn': 'cross_entropy',
    'metric': 'f1_score'
}

# Initialize the AI engine
def init_ai_engine():
    torch.manual_seed(AI_ENGINE_CONFIG['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return AI_ENGINE_CONFIG
