import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
import argparse

from GlossDataset import GlossDataset, NoiseWrapper
from EarlyStop import EarlyStop
from Models import RNN, LSTM, LSTMAttention

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('noise', default=0.1, nargs='?', type=float)
args = parser.parse_args()

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 98 # 7 landmarks for upper body and 21 for each hand for a total of
                # 49 landmarks * 2 x/y positions for each
sequence_length = 50 # 25 fps, assuming about two seconds per video
num_layers = 5
hidden_size = 128
learning_rate = 1e-4
batch_size = 64
num_epochs = 1000
dropout = 0.2

# Logging
now = datetime.now()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
logging_filename = f'log-train-{now.hour}-{now.minute}.log'
if os.path.exists(logging_filename):
    os.remove(logging_filename)
file_handler = logging.FileHandler(logging_filename)

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info('====================')
logger.info(f'STARTING TRAIN {now.hour}:{now.minute}')
logger.info('====================\n')

logger.info(f'Using device {"cuda" if torch.cuda.is_available() else "cpu"}')

logger.info('Network params:')
logger.info(f'\tInput size: {input_size}')
logger.info(f'\tNum layers: {num_layers}')
logger.info(f'\tHidden size: {hidden_size}')
logger.info(f'\tLearning rate: {learning_rate}')
logger.info(f'\tDropout: {dropout}')
logger.info(f'\tBatch size: {batch_size}')
logger.info(f'\tNum epochs: {num_epochs}')


def valid(model: nn.Module, test_loader: DataLoader, loss_function, testing=False) -> float:
  """
  Evaluate the model on the validation set.
  
  Params:
    model (nn.Module): Model to evaluate
    test_loader (DataLoader): Testing dataset
    loss_function: Loss function
    
  Returns:
    (float) Validation loss
    (float) Validation accuracy
  """

  with torch.no_grad():
    n_correct = 0
    n_samples = 0
    total_loss = 0
    total_steps = 0
    model.eval()
    for landmarks, labels, ids in test_loader:
        # make prediction
        landmarks = landmarks.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(landmarks)

        # calculate loss
        loss = loss_function(outputs, labels)
        total_loss += loss.item()
        total_steps += 1

        # statistics
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        if testing:
           logger.info(f'{labels}')
           logger.info(f'{predicted}')
           logger.info(f'{ids}')

    # print stats
    acc = 100.0 * n_correct / n_samples
    return total_loss / total_steps, acc

stopper = EarlyStop(num_epochs) # Setting num_epochs for EarlyStop effectively turns off early stopping

# Assemble data and model
logger.info('Building GlossDataset...')
gloss_data = GlossDataset('processed-videos-filtered.csv', 'asl-data', sequence_length)
logger.info(f'Number of classes: {gloss_data.num_gestures} (random chance accuracy: {100/gloss_data.num_gestures:.2f}%)')
train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(gloss_data, [0.7, 0.2, 0.1])
train_dataset = NoiseWrapper(train_dataset, noise_level=args.noise)

logger.info('Building model...')
num_classes = gloss_data.num_gestures
#model = LSTM(input_size, hidden_size, num_layers, num_classes, dropout).to(device)
model = LSTMAttention(input_size, hidden_size, num_layers, num_classes, dropout).to(device)
#model = RNN(input_size, hidden_size, num_layers, num_classes, batch_size).to(device)

logger.info('Building DataLoaders...')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)
valid_loader = DataLoader(valid_dataset, batch_size=1)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_model = None
highest_accu = 0

# Train the model
logger.info('Beginning training...')
n_total_steps = len(train_loader)
training_data = []
for epoch in range(num_epochs):
    n_correct = 0
    n_samples = 0
    model.train()
    for i, (landmarks, labels, ids) in enumerate(train_loader):
        # Shape: [N, 50, 98]
        landmarks = landmarks.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(landmarks)
        loss = criterion(outputs, labels)

        # Statistics
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_accu = n_correct / n_samples * 100.0
    valid_loss, valid_accu = valid(model, test_loader, criterion)

    if valid_accu > highest_accu:
       highest_accu = valid_accu
       best_model = model.state_dict()

    logger.info(f'Epoch [{epoch+1}/{num_epochs}]:')
    logger.info(f'\tTraining loss:     {loss.item():.4f}')
    logger.info(f'\tTraining accuracy: {train_accu:.4f}%')
    logger.info(f'\tTesting accuracy:  {valid_accu:.4f}%\n')
    if stopper.early_stop(valid_loss):
       logger.info('Stopping early')
       break
    
# Testing set
testing_loss, testing_accu = valid(best_model, valid_loader, criterion, testing=True)
logger.info(f'Testing accuracy: {testing_accu:.4f}%')

logger.info('Saving best model...')
torch.save(best_model, './weights/best-asl-weights.pth')
logger.info('done.')
