import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input, hidden_layers, output) -> None:
        super(MLP, self).__init__()

        self.fc = nn.ModuleList()
        for layer in hidden_layers:
            self.fc.append(nn.Linear(input,layer))
            self.fc.append(nn.BatchNorm1d(layer))
            self.fc.append(nn.ReLU())
            # self.fc.append(nn.Dropout(0.5))
            input = layer
        self.fc.append(nn.Linear(hidden_layers[-1],output))
    
    def forward(self, x):
        out = x 
        for layers in self.fc:
            out = layers(out)
        return out