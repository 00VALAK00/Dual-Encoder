import logging.config
import logging
import yaml
from pathlib import Path

main_dir= Path(__file__).parent.absolute()

config_file_path = main_dir / "config.yaml"
with open(config_file_path, 'r') as conf_file:
    config = yaml.safe_load(conf_file.read())
    logging.config.dictConfig(config)


console_logger=logging.getLogger()
logger = logging.getLogger("my_module")
