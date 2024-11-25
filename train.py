import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import logging

from GlossDataset import GlossDataset
from EarlyStop import EarlyStop
from Models import RNN, LSTM


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 98 # 7 landmarks for upper body and 21 for each hand for a total of
                # 49 landmarks * 2 x/y positions for each
sequence_length = 50 # 25 fps, assuming about two seconds per video
num_layers = 4
hidden_size = 128
learning_rate = 1e-5
batch_size = 16
num_epochs = 100

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('log-train.log')

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f'Using device {"cuda" if torch.cuda.is_available() else "cpu"}')

logger.info('Network params:')
logger.info(f'\tInput size: {input_size}')
logger.info(f'\tNum layers: {num_layers}')
logger.info(f'\tHidden size: {hidden_size}')
logger.info(f'\tLearning rate: {learning_rate}')
logger.info(f'\tBatch size: {batch_size}')
logger.info(f'\tNum epochs: {num_epochs}')

logger.info('\n==============')
logger.info('STARTING TRAIN')
logger.info('==============\n')

def valid(model: nn.Module, test_loader: DataLoader, loss_function) -> float:
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
    for landmarks, labels in test_loader:
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

    # print stats
    acc = 100.0 * n_correct / n_samples
    logger.info(f'Valid acc: {acc:.4f}%')
    logger.info(f'Valid loss: {total_loss / total_steps:.4f}')
    return total_loss / total_steps, acc

stopper = EarlyStop(5)

# Assemble data and model
logger.info('Building GlossDataset...')
gloss_data = GlossDataset('processed-videos.csv', 'asl-data', sequence_length)
train_dataset, test_dataset = torch.utils.data.random_split(gloss_data, [0.8, 0.2])

logger.info('Building model...')
num_classes = gloss_data.num_gestures
model = RNN(input_size, hidden_size, num_layers, num_classes, batch_size).to(device)

logger.info('Building DataLoaders...')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

model_train_data = pd.DataFrame(columns=['epoch', 'train-accu', 'train-loss', 'valid-accu', 'valid-loss'])

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
logger.info('Beginning training...')
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    n_correct = 0
    n_samples = 0
    model.train()
    for i, (landmarks, labels) in enumerate(train_loader):
        # origin shape: [N, 1, 10, 98]
        # resized: [N, 50, 94]
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

    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    logger.info(f'\nTraining ccuracy: {train_accu:.4f}')
    logger.info(f'\nTesting ccuracy: {valid_accu:.4f}')
    if stopper.early_stop(valid_loss):
       logger.info('Stopping early')
       break

    model_train_data.loc[len(model_train_data)] = [epoch, train_accu, loss.item(), valid_accu, valid_loss]

logger.info('Saving model...')
torch.save(model.state_dict(), 'asl-weights.pth')
logger.info('done.')
