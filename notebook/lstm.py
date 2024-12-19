import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # Input gate
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2c = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Forget gate
        self.f2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Output gate
        self.o2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Cell state
        self.c2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Output layer
        self.h2y = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        h, c = hidden
        combined = torch.cat((x, h), 1)
        
        i_t = torch.sigmoid(self.i2h(combined))
        f_t = torch.sigmoid(self.f2h(combined))
        o_t = torch.sigmoid(self.o2h(combined))
        c_tilde = torch.tanh(self.c2h(combined))
        
        c = f_t * c + i_t * c_tilde
        h = o_t * torch.tanh(c)
        
        y_hat = self.h2y(h)
        return y_hat, (h, c)
    
    def init_hidden(self):
        return (torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size))
    
# Create a model
input_size = 10
hidden_size = 20
output_size = 5

lstm = LSTM(input_size, hidden_size, output_size)
print(lstm)

x = torch.randn(1, input_size)
hidden = lstm.init_hidden()
y_hat, hidden = lstm(x, hidden)

print(y_hat)
print(hidden)