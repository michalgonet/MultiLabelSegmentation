import json
from types import SimpleNamespace

from classes import Config


def get_config(path_config: str) -> Config:
    with open(path_config) as file_config:
        config_fields = json.load(file_config, object_hook=lambda d: SimpleNamespace(**d))

    return Config(
        train_data_path=config_fields.Paths.training_data,
        train_label_path=config_fields.Paths.training_labels,
        test_data_path=config_fields.Paths.test_data
    )
