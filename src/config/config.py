# AI Engine Configuration

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration
INPUT_DIM = 10
HIDDEN_DIM = 20
OUTPUT_DIM = 5

# Training configuration
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# Optimizer configuration
OPTIMIZER = 'adam'

# Loss function configuration
LOSS_FN = 'cross_entropy'

# Metric configuration
METRIC = 'f1_score'

# Data configuration
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'

# Logging configuration
LOG_DIR = 'logs'
LOG_FILE = 'ai_engine.log'

# Seed configuration
SEED = 42

# AI Engine configuration dictionary
AI_ENGINE_CONFIG = {
    'device': DEVICE,
    'input_dim': INPUT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'output_dim': OUTPUT_DIM,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'optimizer': OPTIMIZER,
    'loss_fn': LOSS_FN,
    'metric': METRIC,
    'train_data_path': TRAIN_DATA_PATH,
    'test_data_path': TEST_DATA_PATH,
    'log_dir': LOG_DIR,
    'log_file': LOG_FILE,
    'seed': SEED
}
