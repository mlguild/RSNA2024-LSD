from collections import defaultdict
import pathlib
from dataclasses import dataclass
from enum import StrEnum
from re import split
from typing import Any, Callable, Dict, Optional

import h5py
import pandas as pd
import PIL.Image
from gate.data import download_kaggle_dataset, unzip_file
from rich import print
from rich.traceback import install
from torch.utils.data import DataLoader, Dataset
import io

install()


class DatasetNames(StrEnum):
    ISIC_2024 = "isic-2024-challenge"


class SplitNames(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


ISIC2024_COUNT = 401064

METADATA_KEYS = set(
    [
        "isic_id",
        "target",
        "patient_id",
        "clin_size_long_diam_mm",
        "image_type",
        "tbp_tile_type",
        "tbp_lv_A",
        "tbp_lv_Aext",
        "tbp_lv_B",
        "tbp_lv_Bext",
        "tbp_lv_C",
        "tbp_lv_Cext",
        "tbp_lv_H",
        "tbp_lv_Hext",
        "tbp_lv_L",
        "tbp_lv_Lext",
        "tbp_lv_areaMM2",
        "tbp_lv_area_perim_ratio",
        "tbp_lv_color_std_mean",
        "tbp_lv_deltaA",
        "tbp_lv_deltaB",
        "tbp_lv_deltaL",
        "tbp_lv_deltaLB",
        "tbp_lv_deltaLBnorm",
        "tbp_lv_eccentricity",
        "tbp_lv_location",
        "tbp_lv_location_simple",
        "tbp_lv_minorAxisMM",
        "tbp_lv_nevi_confidence",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
        "tbp_lv_perimeterMM",
        "tbp_lv_radial_color_std_max",
        "tbp_lv_stdL",
        "tbp_lv_stdLExt",
        "tbp_lv_symm_2axis",
        "tbp_lv_symm_2axis_angle",
        "tbp_lv_x",
        "tbp_lv_y",
        "tbp_lv_z",
        "attribution",
        "copyright_license",
        "iddx_full",
        "iddx_1",
        "tbp_lv_dnn_lesion_confidence",
    ]
)


@dataclass
class MetadataInfo:
    name: str
    importance: int


class Importance:
    HIGH = 1  # should be used in first string of experiments, MUST HAVEs
    MEDIUM = 2  # could offer additional modelling power, to be used in later experiments, MAY HAVEs
    LOW = 3  # unlikely to offer additional modelling power, to be used in later experiments, ???
    VERY_LOW = 4  # very unlikely to offer additional modelling power, not be used


METADATA_KEYS_TO_USE = [
    MetadataInfo(name="isic_id", importance=Importance.HIGH),
    MetadataInfo(name="target", importance=Importance.HIGH),
    MetadataInfo(name="patient_id", importance=Importance.VERY_LOW),
    MetadataInfo(name="clin_size_long_diam_mm", importance=Importance.MEDIUM),
    MetadataInfo(
        name="image_type",
        importance=Importance.MEDIUM,
    ),
    MetadataInfo(name="tbp_tile_type", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_A", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Aext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_B", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Bext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_C", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Cext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_H", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Hext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_L", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Lext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_areaMM2", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_area_perim_ratio", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_color_std_mean", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaA", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaB", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaL", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaLB", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaLBnorm", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_eccentricity", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_location", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_location_simple", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_minorAxisMM", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_nevi_confidence", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_norm_border", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_norm_color", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_perimeterMM", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_radial_color_std_max", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_stdL", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_stdLExt", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_symm_2axis", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_symm_2axis_angle", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_x", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_y", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_z", importance=Importance.MEDIUM),
    MetadataInfo(name="attribution", importance=Importance.VERY_LOW),
    MetadataInfo(name="copyright_license", importance=Importance.VERY_LOW),
    MetadataInfo(name="iddx_full", importance=Importance.HIGH),
    MetadataInfo(name="iddx_1", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_dnn_lesion_confidence", importance=Importance.HIGH),
]


@dataclass
class DataItem:
    image: PIL.Image.Image
    labels: Dict[str, Any]


ISIC_ID = "isic_id"


class ISIC2024Dataset(Dataset):

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        split_name: SplitNames | str = SplitNames.TRAIN,
        transform: Optional[Callable] = None,
        importance_level_labels: Importance = Importance.HIGH,
    ):
        super().__init__()
        if isinstance(root_dir, str):
            root_dir = pathlib.Path(root_dir)

        self.root_dir = root_dir
        self.transform = transform
        self.download_extract_data()
        self.file_count_after_download_and_extract = ISIC2024_COUNT

        if split_name == SplitNames.TRAIN or split_name == SplitNames.VAL:
            target_split_name = "train"
            self.data = self.root_dir / DatasetNames.ISIC_2024 / f"{target_split_name}-image.hdf5"
            self.data = h5py.File(self.data, "r")
            self.metadata_path = (
                self.root_dir / DatasetNames.ISIC_2024 / f"{target_split_name}-metadata.csv"
            )
            self.metadata = pd.read_csv(self.metadata_path)
        elif split_name == SplitNames.TEST:
            target_split_name = "test"
            self.data = self.root_dir / DatasetNames.ISIC_2024 / f"{target_split_name}-image.hdf5"
            self.data = h5py.File(self.data, "r")
            self.metadata_path = (
                self.root_dir / DatasetNames.ISIC_2024 / f"{target_split_name}-metadata.csv"
            )
            self.metadata = pd.read_csv(self.metadata_path)

        self.label_keys_to_use = [
            item.name for item in METADATA_KEYS_TO_USE if item.importance <= importance_level_labels
        ]

    def download_extract_data(self):
        # use kaggle to download the isic-2024-challenge dataset
        # kaggle competitions download -c isic-2024-challenge
        download_kaggle_dataset(
            dataset_name=DatasetNames.ISIC_2024,
            dataset_path=DatasetNames.ISIC_2024,
            target_dir_path=self.root_dir,
            is_competition=True,
            unzip=True,
            file_count_after_download_and_extract=4000,
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        labels = {key: self.metadata.iloc[idx][key] for key in self.label_keys_to_use}
        image_bytes = self.data[labels[ISIC_ID]][()]
        image = PIL.Image.open(io.BytesIO(image_bytes))

        if self.transform:
            image = self.transform(image)

        return DataItem(image=image, labels=labels)


if __name__ == "__main__":
    # tmp_data_dir = "/mnt/nvme-fast0/datasets/"
    # dataset = Isic2024Dataset(root_dir=tmp_data_dir)
    # print(dataset)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # print(dataloader)
    # for data in dataloader:
    #     print(data)
    #     break
    # load in an hdf5 file and iterate the first 10 entries
    # import h5py
    # import numpy as np
    # from PIL import Image
    # import io

    # hdf5_file_path = "/mnt/nvme-fast0/datasets/isic-2024-challenge/train-image.hdf5"
    # with h5py.File(hdf5_file_path, "r") as f:
    #     for key, value in f.items():
    #         print(key)
    #         # inspect an HDF5 dataset object stored in value
    #         # cast it as a numpy array
    #         image_bytes = value[()]
    #         # convert the numpy array to a PIL image
    #         image = Image.open(io.BytesIO(image_bytes))
    #         image.save("dummy.png")
    #         break

    # csv_file_path = "/mnt/nvme-fast0/datasets/isic-2024-challenge/train-metadata.csv"

    # # iterate the first 10 entries in the metadata csv file
    # import pandas as pd

    # metadata = pd.read_csv(csv_file_path)
    # # show the full verbose first 10 entries
    # # drop columns that are not available for all entries
    # metadata = metadata.dropna(axis=1, how="any")
    # # show all columns keys
    # print(metadata.columns)

    tmp_data_dir = "/mnt/nvme-fast0/datasets/"
    from gate.data.image.classification.imagenet1k import StandardAugmentations
    from tqdm.auto import tqdm

    transforms = [StandardAugmentations()]
    dataset = ISIC2024Dataset(root_dir=tmp_data_dir)

    print(dataset)
    image_size_freq = defaultdict(int)
    for item in tqdm(dataset):
        # print(item)
        image_size = item.image.size
        image_size_freq[image_size] += 1

    print(image_size_freq)
