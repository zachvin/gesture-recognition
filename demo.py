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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def valid(model: nn.Module, test_loader: DataLoader) -> float:
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
    model.eval()
    for landmarks, labels in test_loader:
        # make prediction
        landmarks = landmarks.reshape(-1, 50, 98).to(device)
        labels = labels.to(device)
        outputs = model(landmarks)

        # statistics
        _, predicted = torch.max(outputs.data, 1)

        print(f'Correct label: {labels.item()}')
        print(f'Predicted label: {predicted.item()}\n')

# Assemble data and model
gloss_data = GlossDataset('processed-videos-demo.csv', 'asl-data', 50)
num_classes = gloss_data.num_gestures
model = LSTMAttention(98, 128, 5, 39, dropout=0).to(device)
model.load_state_dict(torch.load('weights/10pct39cl.pth', map_location=device, weights_only=True))

test_loader = DataLoader(gloss_data, batch_size=1)
valid(model, test_loader)