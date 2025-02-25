import logging
from .config_manager import ConfigManager

conf = ConfigManager('conf.json')

logging.basicConfig(
    level=logging.DEBUG,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=conf.get('log_file')
)

def get_logger(name):
    return logging.getLogger(name)