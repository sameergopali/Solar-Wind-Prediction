from pathlib import Path
from config_parser import ConfigParser
from dataloader import get_dataset, get_avg_dataloader
import os
import torch
import torch.nn as nn
from models.mlp import  MLP
from torch import optim
from trainer import Trainer
from utils import set_seed
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main(config):
    set_seed(1)
    logger = config.get_logger('main')
    base_path = os.path.dirname(__file__)
    logger.debug(base_path)
    learning_rate =  config.config['trainer']['lr']
    dataX_path = os.path.join(os.path.join(base_path,'data/processed/mean_ACE.pkl'))
    dataY_path = os.path.join(os.path.join(base_path,'data/processed/ACE_Y_mean.pkl'))
    dataX, dataY = get_dataset(dataX_path,dataY_path)
    dataloaders = get_avg_dataloader(dataX,dataY, batchsize=128)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    logger.debug(f'Using {device=}')
    modelcfg= config.config['model']
    model = MLP(input=modelcfg['input_size'],output=modelcfg['output_size'], hidden_layers=modelcfg['hidden_layers']).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(model, parameters)
    criterion =  nn.MSELoss()
    optimizer  = optim.Adam(model.parameters(), lr = learning_rate )
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, config=config,device=device, data_loader=dataloaders)
    trainer.train()
    test(model, dataloaders['test_loader'],device, logger)

def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n_samples = len(y_true)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features )
    return adjusted_r2
def test(net,test_loader,device):
    y_pred = []
    y_true = []
    net.train(False)
    net.eval()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets
        y_pred.extend(net(inputs).cpu().data.numpy())
        y_true.extend(targets.numpy())

    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", mean_squared_error(y_true, y_pred,squared=False))
    print("R^2:", r2_score(y_true, y_pred))
    print('adjusted r2',adjusted_r2_score(y_true, y_pred, 11))

if __name__ == '__main__':
    args = {
        'name':'Updated_data',
        'trainer':{
            'save_dir':'save',
            'epochs': 2000,
            'lr':10e-3
        },
        'model':{
          'type':'MLP',
          'input_size':11,
          'output_size':1,
          'hidden_layers':[1024,512,256,64],
    
        }
    }
    config =  ConfigParser(run_id='mlp',**args)
    main(config)