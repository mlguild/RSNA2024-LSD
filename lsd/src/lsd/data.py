import concurrent.futures
import io
import multiprocessing as mp
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Callable, Dict, Iterator, List, Optional

import h5py
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torchvision.transforms as T
from gate.data import download_kaggle_dataset
from gate.data.image.classification.imagenet1k import StandardAugmentations
from rich import print
from rich.traceback import install
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

install()


class DatasetNames(StrEnum):
    ISIC_2024 = "isic-2024-challenge"


class SplitNames(StrEnum):
    TRAIN = "train"
    VAL = "val"
    DEVTEST = "devtest"
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
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    VERY_LOW = 4


METADATA_KEYS_TO_USE = [
    MetadataInfo(name="isic_id", importance=Importance.HIGH),
    MetadataInfo(name="target", importance=Importance.HIGH),
    MetadataInfo(name="patient_id", importance=Importance.VERY_LOW),
    MetadataInfo(name="clin_size_long_diam_mm", importance=Importance.MEDIUM),
    MetadataInfo(name="image_type", importance=Importance.MEDIUM),
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
    MetadataInfo(
        name="tbp_lv_radial_color_std_max", importance=Importance.MEDIUM
    ),
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
    MetadataInfo(
        name="tbp_lv_dnn_lesion_confidence", importance=Importance.HIGH
    ),
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
        return_samples_as_dict: bool = False,
    ):
        super().__init__()
        if isinstance(root_dir, str):
            root_dir = pathlib.Path(root_dir)

        self.root_dir = root_dir
        self.transform = transform
        self.download_extract_data()
        self.file_count_after_download_and_extract = ISIC2024_COUNT
        self.return_samples_as_dict = return_samples_as_dict

        self.split_name = split_name

        if split_name in {
            SplitNames.TRAIN,
            SplitNames.VAL,
            SplitNames.DEVTEST,
        }:
            target_split_name = "train"
        elif split_name == SplitNames.TEST:
            target_split_name = "test"
        else:
            raise ValueError(f"Invalid split name: {split_name}")

        self.data = (
            self.root_dir
            / DatasetNames.ISIC_2024
            / f"{target_split_name}-image.hdf5"
        )
        self.data = h5py.File(self.data, "r")
        self.metadata_path = (
            self.root_dir
            / DatasetNames.ISIC_2024
            / f"{target_split_name}-metadata.csv"
        )
        self.metadata = pd.read_csv(self.metadata_path)

        if split_name in {
            SplitNames.TRAIN,
            SplitNames.VAL,
            SplitNames.DEVTEST,
        }:
            self.metadata = self._split_metadata()

        self.label_keys_to_use = [
            item.name
            for item in METADATA_KEYS_TO_USE
            if item.importance <= importance_level_labels
        ]

        print("Creating class to index mapper...")
        self.class_indices = self.create_class_indices()

    def _split_metadata(self):
        """Split the training metadata into train, val, and devtest"""
        train_ratio, val_ratio, devtest_ratio = 0.9, 0.05, 0.05
        train_end = int(train_ratio * len(self.metadata))
        val_end = int((train_ratio + val_ratio) * len(self.metadata))

        if self.split_name == SplitNames.TRAIN:
            return self.metadata[:train_end]
        elif self.split_name == SplitNames.VAL:
            return self.metadata[train_end:val_end]
        elif self.split_name == SplitNames.DEVTEST:
            return self.metadata[val_end:]
        else:
            raise ValueError(
                f"Invalid split name for splitting metadata: {self.split_name}"
            )

    def create_class_indices(self):
        class_indices = defaultdict(list)

        def process_row(args):
            idx, row = args
            return int(row["target"]), idx

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=mp.cpu_count()
        ) as executor:
            future_to_row = {
                executor.submit(process_row, (idx, row)): idx
                for idx, row in self.metadata.iterrows()
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_row),
                total=len(self.metadata),
                desc="Processing rows",
            ):
                target, idx = future.result()
                class_indices[target].append(idx)

        return class_indices

    def download_extract_data(self):
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
        labels = {
            key: self.metadata.iloc[idx][key] for key in self.label_keys_to_use
        }
        image_bytes = self.data[labels[ISIC_ID]][()]
        image = PIL.Image.open(io.BytesIO(image_bytes))

        if self.transform:
            image = self.transform(image)

        output = DataItem(image=image, labels=labels)

        if self.return_samples_as_dict:
            output = asdict(output)
        return output


class BalancedBatchSampler(Sampler):
    def __init__(
        self,
        dataset: ISIC2024Dataset,
        batch_size: int,
        class_ratios: List[float],
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_ratios = class_ratios
        self.num_classes = len(class_ratios)

        print(
            f"Initializing BalancedBatchSampler with batch size {batch_size} and class ratios {class_ratios}"
        )

        # Create class indices based on the Subset
        self.class_indices = [[] for _ in range(self.num_classes)]
        for idx in range(len(self.dataset)):
            label = self.dataset.metadata.iloc[idx]["target"]
            self.class_indices[int(label)].append(idx)

        # Print class distribution
        for class_idx, indices in enumerate(self.class_indices):
            print(f"Class {class_idx}: {len(indices)} samples")

        # Calculate number of samples for each class in a batch
        self.samples_per_class = [
            max(1, int(batch_size * ratio)) for ratio in class_ratios
        ]

        # Adjust if total samples exceed batch size
        while sum(self.samples_per_class) > batch_size:
            max_class = self.samples_per_class.index(
                max(self.samples_per_class)
            )
            self.samples_per_class[max_class] -= 1

        print(f"Samples per class in each batch: {self.samples_per_class}")

        # Calculate the number of batches
        self.num_batches_per_epoch = min(
            len(indices) // samples
            for indices, samples in zip(
                self.class_indices, self.samples_per_class
            )
        )
        print(f"Number of batches per epoch: {self.num_batches_per_epoch}")
        self.batch_count = 0
        # Start off by reshuffling indices
        self._reshuffle_indices()

    def _reshuffle_indices(self):
        """Reshuffle class indices for a new epoch."""
        # print("Reshuffling indices for a new epoch.")
        self.remaining_indices = [
            np.random.permutation(indices).tolist()
            for indices in self.class_indices
        ]

    def __iter__(self) -> Iterator[List[int]]:
        while True:  # Infinite loop to keep yielding batches
            if self.batch_count >= self.num_batches_per_epoch:
                self._reshuffle_indices()

            batch = []
            for class_idx, num_samples in enumerate(self.samples_per_class):
                class_indices = self.remaining_indices[class_idx]

                if len(class_indices) < num_samples:
                    # print(
                    #     f"Not enough samples in class {class_idx}. Reshuffling the indices."
                    # )
                    # Reshuffle within the class if we run out of samples mid-epoch
                    self.remaining_indices[class_idx] = np.random.permutation(
                        self.class_indices[class_idx]
                    ).tolist()
                    class_indices = self.remaining_indices[class_idx]

                sampled_indices = class_indices[:num_samples]
                # print(
                #     f"Sampled indices for class {class_idx}: {sampled_indices}"
                # )
                batch.extend(sampled_indices)
                self.remaining_indices[class_idx] = class_indices[
                    num_samples:
                ]  # Remove sampled indices

            np.random.shuffle(batch)
            yield batch
            self.batch_count += 1

    def __len__(self) -> int:
        return self.num_batches_per_epoch


def main():
    tmp_data_dir = "/mnt/nvme-fast0/datasets/"
    transforms = [T.Resize(224), StandardAugmentations(), T.ToTensor()]
    dataset = ISIC2024Dataset(
        root_dir=tmp_data_dir,
        transform=T.Compose(transforms),
        return_samples_as_dict=True,
    )

    print(f"Total dataset size: {len(dataset)}")
    print("Class distribution:")
    for class_idx, indices in dataset.class_indices.items():
        print(f"Class {class_idx}: {len(indices)} samples")

    batch_size = 32
    class_ratios = [0.8, 0.2]  # 80% class 0, 20% class 1
    balanced_sampler = BalancedBatchSampler(dataset, batch_size, class_ratios)

    print(f"Number of batches: {len(balanced_sampler)}")

    dataloader = DataLoader(
        dataset,
        batch_sampler=balanced_sampler,
        num_workers=16,
        pin_memory=True,
    )

    label_frequencies = defaultdict(int)
    num_batches_to_check = min(
        10, len(balanced_sampler)
    )  # Adjust this to check more or fewer batches

    for i, item in enumerate(
        tqdm(dataloader, total=num_batches_to_check, desc="Checking batches")
    ):
        if i >= num_batches_to_check:
            break
        for label in item["labels"]["target"]:
            label_frequencies[int(label)] += 1

        print(f"Batch {i+1} label frequencies: {dict(label_frequencies)}")
        label_frequencies.clear()  # Reset for next batch

    print("Finished checking batches.")


if __name__ == "__main__":
    main()
