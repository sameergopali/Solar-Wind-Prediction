import logging
from pathlib import Path
from datetime import datetime
from logger import setup_logging
from utils import read_json

class ConfigParser:
    def __init__(self, run_id=None, **kwargs, ):
    
        self._config = kwargs
        
        save_dir = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']

        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        self._log_dir = save_dir/'log'/exper_name/run_id
        self._model_dir = save_dir/'model'/exper_name/run_id

        exist_ok = run_id == ''
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.DEBUG,
            1: logging.INFO,
            2: logging.WARNING
        }

    def get_logger(self,name,verbosity=0):
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger 

    @classmethod
    def from_args(cls, args):
      
        return cls(args)
    
    @property
    def model_dir(self):
        return self._model_dir
    
    @property 
    def log_dir(self):
        return self._log_dir
    
    @property
    def config(self):
        return self._config