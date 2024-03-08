import logging

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd, Spacingd, \
    RandSpatialCropd, RandFlipd, NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd

from classes import Config, ConvertToMultiChannelBasedOnBratsClassesd

log = logging.getLogger(__name__)


def get_train_transform(config: Config) -> Compose:
    return Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=config.spacing, mode=("bilinear", "nearest"), ),
            RandSpatialCropd(keys=["image", "label"], roi_size=config.roi_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=config.aug_rand_flip_prob, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=config.aug_rand_flip_prob, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=config.aug_rand_flip_prob, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image",
                                factors=config.aug_rand_scale_intensity[0],
                                prob=config.aug_rand_scale_intensity[1]),
            RandShiftIntensityd(keys="image",
                                offsets=config.aug_rand_shift_intensity[0],
                                prob=config.aug_rand_shift_intensity[1]),
        ]
    )


def get_val_transform(config: Config) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=config.spacing, mode=("bilinear", "nearest"), ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


def training(config):
    get_train_transform(config)
    logging.info('Train transform ready')
    get_val_transform(config)
    logging.info('Validation transform ready')
