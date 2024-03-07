from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class Config:
    train_data_path: str
    train_label_path: str
    test_data_path: str


class Flag(Enum):
    prepare_data = auto()
    training = auto()
    testing = auto()
