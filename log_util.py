import sys
import logging


def basic_log_config():
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
        level=logging.INFO,
    )
