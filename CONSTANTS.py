import torch

# Define global params for training
LEARNING_RATE = 0.01
EPOCHS = 50
EMBEDDING_SIZE = 50
DEBUG_LENGTH = 10000
HIDDEN_SIZE = 1000
UNIQUE_WORDS = 2000
SOS_token = 1
EOS_token = 2

# Set this to true if testing
DEBUG = True

# Detect if Cuda should be used
use_cuda = torch.cuda.is_available()