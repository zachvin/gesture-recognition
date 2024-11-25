import torch
import torch.nn as nn

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes,
               batch_size):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_classes = num_classes
    self.batch_size = batch_size
    self.device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

    # RNN takes tensor of shape (batch_size, sequence_length, input_size)
    # (N, 30, 90)
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    # classifier -- uses final hidden state as input, outputs probability of
    # each class
    self.fc = nn.Linear(self.hidden_size, self.num_classes)
    self.softmax = nn.LogSoftmax()

  def forward(self, x):
    # x = (N, 30, 90) = (batch_size, sequence_length, input_size)
    # h_0 = (2, N, 128) = (num_layers, batch_size, hidden_size)
    h_0 = torch.zeros(self.num_layers, self.batch_size,
                              self.hidden_size).to(torch.float32).to(self.device)
        
    if self.batch_size == 1:
      h_0 = h_0.reshape((self.num_layers, self.hidden_size))

    # get RNN last layer output. last hidden layer is no longer necessary
    x = x.to(torch.float32)
    output, h_n = self.rnn(x, h_0)

    # output = (batch_size, sequence_length, hidden_size) = (N, 30, 90)
    output = output[-1, :] # output of last layer for each batch sequence

    # output = (batch_size, hidden_size)
    output = self.fc(output)
    output = self.softmax(output)

    # output = (batch_size, num_classes)
    return output
  
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(LSTM, self).__init__()

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.label = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    # x: shape [batch_size, seq_length, input_size]
    lstm_out, _ = self.lstm(x)  # lstm_out: shape [batch_size, seq_length, hidden_size]
    out = self.label(lstm_out[:, -1, :])  # use the last time step's hidden state

    return out