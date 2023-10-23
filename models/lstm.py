import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fc_layers,  device, dropout=0) -> None:
        super(LSTM, self).__init__() 
        self.device=device
        self.hidden_size = hidden_size
        self.num_layers =  num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first =True, dropout=dropout)
        self.activation = nn.ReLU()

        self.fc = nn.ModuleList()
        for layer in fc_layers:
            self.fc.append(nn.Linear(hidden_size,layer))
            self.fc.append(nn.BatchNorm1d(layer))
            self.fc.append(nn.ReLU())
            #self.fc.append(nn.Dropout(0.5))
            hidden_size = layer
        self.fc.append(nn.Linear(fc_layers[-1],1))
       
    
    def forward(self,x):
        h0  =  torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device).requires_grad_()
        c0  =  torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device).requires_grad_()

        out, (hout,_) =  self.lstm(x,(h0,c0))
        #out = self.activation(out[:,-1,:])
       
        out = out[:,-1,:]
        for layers in self.fc:
            out = layers(out)
        #out = self.fc(out)
        return out
        #out = self.fc(out)
