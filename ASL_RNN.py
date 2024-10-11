import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 98 # 7 landmarks for upper body and 21 for each hand for a total of
                # 49 landmarks * 2 x/y positions for each
sequence_length = 50 # 25 fps, assuming about two seconds per video
num_layers = 2
hidden_size = 128
num_classes = 5 # number of signs
learning_rate = 0.0001
batch_size = 16
num_epochs = 100

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
    self.landmark_labels = pd.read_csv(annotations_file, dtype={'id': 'object'})
    self.landmark_dir = landmark_dir
    self.sequence_length = sequence_length
    with open('gloss_to_id.json', 'r') as f:
        self.gloss_to_id = json.loads(json.load(f))

    self.landmark_labels['gloss'] = self.landmark_labels['gloss'].apply(lambda x : self.gloss_to_int[x])

  def __len__(self):
    return len(self.landmark_labels)

  def __getitem__(self, idx):
    landmark_path = os.path.join(self.landmark_dir, self.landmark_labels.iloc[idx, 0] + '.csv')

    gloss = self.landmark_labels.iloc[idx, 1]
    landmarks = pd.read_csv(landmark_path)
    # pad output to make video long enough
    if landmarks.shape[1] - self.sequence_length > 0:
      delta = landmarks.shape[1] - self.sequence_length
      row = landmarks.iloc[-1]
      for _ in range(delta):
        landmarks.loc[len(landmarks)] = row

    # trim output if it's too long
    landmarks_tensor = torch.tensor(landmarks.iloc[:self.sequence_length].to_numpy().astype('float32'))

    return landmarks_tensor, gloss
  
def valid():
  with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in train_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Valid acc: {acc}%')
   
  
model = RNN(input_size, hidden_size, num_layers, num_classes, batch_size).to(device)

gloss_data = GlossDataset('video-metadata.csv', 'asl-data', sequence_length)
train_loader = DataLoader(gloss_data, batch_size=16, shuffle=True)
test_loader = DataLoader(gloss_data, batch_size=4)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
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

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    loss = valid()
    print(f'\tValid loss: {loss:.5f}')