import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, output_size):
        super(LSTM, self).__init__()

        self.cell_size = cell_size

        self.forget = nn.Linear(input_size + cell_size, cell_size)
        
        self.input = nn.Linear(input_size + cell_size, cell_size)
        self.scale = nn.Linear(input_size + cell_size, cell_size)

        self.hidden = nn.Linear(input_size + cell_size, cell_size)
        self.output = nn.Linear(cell_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        combined = torch.cat((hidden, input), 1)
        
        forget_gate = self.sigmoid(self.forget(combined))

        input_gate = self.sigmoid(self.input(combined)) * self.tanh(self.scale(combined))
        
        cell = cell * forget_gate + input_gate

        hidden = self.sigmoid(self.hidden(combined)) * self.tanh(cell)
        output = self.softmax(self.output(hidden))

        return output, hidden, cell
    
    def initCell(self):
        return torch.zeros(1, self.cell_size)
        
    def initHidden(self):
        return torch.zeros(1, self.cell_size)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)