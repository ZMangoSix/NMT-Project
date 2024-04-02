import torch
from torch import nn

### RNN
# class RNN(nn.Module):
#     def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
#         super().__init__()
#         self.model = nn.RNN(num_inputs, num_hiddens, num_layers, dropout=dropout)

    
#     def forward(self, inputs, H=None):
#         return self.model(inputs, H)


# GRU
class GRU(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.model = nn.GRU(num_inputs, num_hiddens, num_layers, dropout=dropout)

    
    def forward(self, inputs, H=None):
        return self.model(inputs, H)


if __name__ == '__main__':
    pass
