'''This file consists of all the hyperparameters that will be used during training'''

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1000
learning_rate = 1e-3
batch_size = 4