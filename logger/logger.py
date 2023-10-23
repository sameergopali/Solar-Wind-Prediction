import logging
import logging.config
from pathlib import Path
from utils import read_json

def setup_logging(save_dir,default_level=logging.INFO,log_config='logger/logger_config.json'):
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])
        logging.config.dictConfig(config)
    else:
        print(f"Warning: logging configuration file not found in {log_config}")
        logging.basicConfig(level=default_level)
