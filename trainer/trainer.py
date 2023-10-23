from base import BaseTrainer
import torch

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, config,device, data_loader):
        super().__init__(model, criterion, optimizer, config)
        self.device = device
        self.train_loader =  data_loader['train_loader']
        self.val_loader =  data_loader['val_loader']
        
    def _train_epoch(self, epoch):
        self.model.train()
        training_loss = 0 
        for inputs, target in self.train_loader:
            inputs, target = inputs.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(inputs)
            loss = self.criterion(y_pred,torch.unsqueeze(target,dim=1))
            loss.backward()
            self.optimizer.step()
            training_loss += loss.item()
        train_loss = training_loss/len(self.train_loader)
        validation_loss = self._validation_epoch()
        loss  = {'training_loss':train_loss, "validation_loss":validation_loss}
        return loss
    
    def _validation_epoch(self):
        self.model.eval()
        validation_loss  = 0 
        with torch.no_grad():
            for inputs, target in self.val_loader:
                inputs, target = inputs.to(self.device), target.to(self.device)
                y_pred = self.model(inputs)
                loss = self.criterion(y_pred, torch.unsqueeze(target, dim=1))
                validation_loss += loss.item()
        return validation_loss/len(self.val_loader)
