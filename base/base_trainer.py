import torch
from numpy import Inf
from abc import ABC, abstractmethod

class BaseTrainer(ABC):

    def __init__(self,model, criterion,optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer')
       

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_dir = config.model_dir

        self.validation_losses =[]
        self.training_losses = []
        

        cfg_trainer = config.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save = True

    @abstractmethod
    def _train_epoch(self, epoch):
        raise  NotImplementedError

    def train(self):
        best_validation_loss = Inf
        for epoch in range(self.epochs):
            result  = self._train_epoch(epoch)
            training_loss = result['training_loss']
            validation_loss = result['validation_loss']
            if self.save and best_validation_loss > validation_loss:
                best_validation_loss = validation_loss
                best = self.model.state_dict()
            self.training_losses.append(training_loss)
            self.validation_losses.append(validation_loss)
            self.logger.info(f"Epoch #{epoch + 1}\t Training loss :{training_loss}\t Validation loss: {validation_loss}" )
        self._save_best(best)

    def _save_best(self,state_dict):
        state = {
            'state_dict':state_dict,
            'optimizer': self.optimizer.state_dict(),
            'validation_losses' : self.validation_losses,
            'training_losses' : self.training_losses
        }
        filename =  str(self.checkpoint_dir/'model.best.pth')
        torch.save(state, filename)
        self.logger.info("Saving current best: model")


 