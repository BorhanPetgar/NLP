import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2y = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h):
        h = torch.tanh(self.x2h(x) + self.h2h(h))
        y_hat = self.h2y(h)
        
        return y_hat, h
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
# Create a model
input_size = 10
hidden_size = 20
output_size = 5
rnn = RNN(input_size, hidden_size, output_size)
print(rnn)

x = torch.randn(1, 10)
h = rnn.init_hidden()
y_hat, h = rnn(x, h)

print(y_hat)
print(h)