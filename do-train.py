import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='do-train.log', level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device {"cuda" if torch.cuda.is_available() else "cpu"}')


input_size = 98 # 7 landmarks for upper body and 21 for each hand for a total of
                # 49 landmarks * 2 x/y positions for each
sequence_length = 50 # 25 fps, assuming about two seconds per video
num_layers = 2
hidden_size = 128
learning_rate = 0.0001
batch_size = 16
num_epochs = 100

logger.info('Network params:')
logger.info(f'\tInput size: {input_size}')
logger.info(f'\tNum layers: {num_layers}')
logger.info(f'\tHidden size: {hidden_size}')
logger.info(f'\tLearning rate: {learning_rate}')
logger.info(f'\tBatch size: {batch_size}')
logger.info(f'\tNum epochs: {num_epochs}')

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes,
               batch_size):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_classes = num_classes
    self.batch_size = batch_size

    # RNN takes tensor of shape (batch_size, sequence_length, input_size)
    # (N, 30, 90)
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    # classifier -- uses final hidden state as input, outputs probability of
    # each class
    self.fc = nn.Linear(self.hidden_size, self.num_classes)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x):
    # x = (N, 30, 90) = (batch_size, sequence_length, input_size)
    # h_0 = (2, N, 128) = (num_layers, batch_size, hidden_size)
    h_0 = torch.zeros(self.num_layers, x.size(0),
                              self.hidden_size).to(device)

    # get RNN last layer output. last hidden layer is no longer necessary
    output, h_n = self.rnn(x, h_0)

    # output = (batch_size, sequence_length, hidden_size) = (N, 30, 90)
    output = output[:, -1, :] # output of last layer for each batch sequence

    # output = (batch_size, hidden_size)
    output = self.fc(output)
    output = self.softmax(output)

    # output = (batch_size, num_classes)
    return output

class GlossDataset(Dataset):
  def __init__(self, annotations_file, landmark_dir, sequence_length):
    self.gloss_and_id = pd.read_csv(annotations_file, dtype={'id': 'object', 'gloss':'category'})
    cat_map = {gloss_name: gloss_id for gloss_id, gloss_name in enumerate(self.gloss_and_id['gloss'].cat.categories)}
    self.gloss_and_id['gloss'] = self.gloss_and_id['gloss'].replace(cat_map)
    self.gloss_and_id['gloss'] = self.gloss_and_id['gloss'].astype(int)

    self.landmark_dir = landmark_dir
    self.sequence_length = sequence_length
    self.num_gestures = self.gloss_and_id['gloss'].nunique()

  def __len__(self):
    return len(self.gloss_and_id)

  def __getitem__(self, idx):
    landmark_path = os.path.join(self.landmark_dir, self.gloss_and_id['id'].iloc[idx] + '.csv')

    gloss = self.gloss_and_id['gloss'].iloc[idx]

    landmarks = pd.read_csv(landmark_path)
    # pad output to make video long enough
    if landmarks.shape[1] - self.sequence_length > 0:
      delta = landmarks.shape[1] - self.sequence_length
      row = landmarks.iloc[-1]
      for _ in range(delta):
        landmarks.loc[len(landmarks)] = row

    # trim output if it's too long
    landmarks_tensor = torch.tensor(landmarks.iloc[:self.sequence_length].to_numpy().astype('float32'))
    labels_tensor = torch.tensor(gloss.astype('float32'))

    return landmarks_tensor, gloss

def valid(model, test_loader, loss_function):
  with torch.no_grad():
    n_correct = 0
    n_samples = 0
    total_loss = 0
    total_steps = 0
    model.eval()
    for landmarks, labels in test_loader:
        landmarks = landmarks.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(landmarks)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        loss = loss_function(predicted, labels)
        total_loss += loss.item()
        total_steps += 1

    acc = 100.0 * n_correct / n_samples
    logger.info(f'Valid acc: {acc}%')
    logger.info(f'Valid loss: {total_loss / total_steps}')

class EarlyStop:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


stopper = EarlyStop(5)

logger.info('Building GlossDataset...')
gloss_data = GlossDataset('processed-videos.csv', 'asl-data', sequence_length)
train_dataset, test_dataset = torch.utils.data.random_split(gloss_data, [0.8, 0.2])

logger.info('Building model...')
num_classes = gloss_data.num_gestures
model = RNN(input_size, hidden_size, num_layers, num_classes, batch_size).to(device)

logger.info('Building DataLoaders...')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
logger.info('Beginning training...')
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (landmarks, labels) in enumerate(train_loader):
        # origin shape: [N, 1, 10, 98]
        # resized: [N, 50, 94]
        landmarks = landmarks.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(landmarks)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    valid_loss = valid(model, test_loader, criterion)
    if stopper.early_stop(valid_loss):
       logger.info('Stopping early')
       break

logger.info('Saving model...')
torch.save(model.state_dict(), 'ASL_RNN.pth')
logger.info('done.')
