import json
from types import SimpleNamespace

from classes import Config


def get_config(path_config: str) -> Config:
    with open(path_config) as file_config:
        config_fields = json.load(file_config, object_hook=lambda d: SimpleNamespace(**d))

    return Config(
        train_data_path=config_fields.Paths.training_data,
        train_label_path=config_fields.Paths.training_labels,
        test_data_path=config_fields.Paths.test_data,
        aug_rand_flip_prob=config_fields.Augmentation.rand_flip_prob,
        aug_rand_scale_intensity=config_fields.Augmentation.rand_scale_intensity,
        aug_rand_shift_intensity=config_fields.Augmentation.rand_shift_intensity,
        roi_size=config_fields.Data.roi_size,
        spacing=config_fields.Data.spacing
    )
