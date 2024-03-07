import logging

import click
from pathlib import Path

from analysis.config import get_config
from classes import Flag
from log_util import basic_log_config


@click.command()
@click.option('--config-path', type=click.Path(exists=True, dir_okay=False, path_type=Path), default='config.json',
              help='Optional. Path to the configuration JSON. Default: config.json')
@click.option('--flag', type=str, default='prepare_data',
              help='Optional. String flag to start ML pipeline prepare_data/training/testing')
def main(config_path: str, flag: Flag):
    config = get_config(config_path)

    if flag == Flag.prepare_data.name:
        logging.info(f'Start data preparation')
    elif flag == Flag.training.name:
        logging.info(f'Start training')
    elif flag == Flag.testing.name:
        logging.info(f'Start testing')
    else:
        raise KeyError(f'Wrong flag {flag}')


if __name__ == '__main__':
    basic_log_config()
    main()
