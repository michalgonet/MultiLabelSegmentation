from dataclasses import dataclass
from enum import Enum, auto

import torch
from monai.transforms import MapTransform


@dataclass(frozen=True)
class Config:
    train_data_path: str
    train_label_path: str
    test_data_path: str
    aug_rand_flip_prob: float
    aug_rand_scale_intensity: list[float, float]
    aug_rand_shift_intensity: list[float, float]
    roi_size: list[int, int, int]
    spacing: list[float, float, float]


class Flag(Enum):
    prepare_data = auto()
    training = auto()
    testing = auto()


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi-channels based on BRATS classes:
    label 1 is the peritumoral edema,
    label 2 is the GD-enhancing tumor,
    label 3 is the necrotic and non-enhancing tumor core.
    The possible classes are TC (Tumor core), WT (Whole tumor),
    and ET (Enhancing tumor).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            tc_mask = (label == 2) | (label == 3)  # Tumor Core (TC) mask
            wt_mask = (label == 1) | tc_mask  # Whole Tumor (WT) mask
            et_mask = (label == 2)  # Enhancing Tumor (ET) mask

            # Convert masks to float and stack as channels
            multi_channel_label = torch.stack([tc_mask.float(), wt_mask.float(), et_mask.float()], dim=0)
            d[key] = multi_channel_label
        return d
