"""
Datasets for training and testing.
"""
import logging
import os
from pathlib import Path
from typing import Optional
from typing import Union

import lightning as L
import torch
import torchaudio
from torch.utils.data import DataLoader

from timbreremap.np import OnsetFrames

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class OnsetFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset that returns pairs of onset features with full features
    """

    def __init__(
        self,
        onset_features: torch.Tensor,  # Onset features
        full_features: torch.Tensor,  # Full features
        weight: Optional[torch.Tensor] = None,  # Feature weighting
        onset_ref: Optional[torch.Tensor] = None,  # Onset feature values for reference
    ):
        super().__init__()
        self.onset_features = onset_features
        self.full_features = full_features
        assert self.onset_features.shape[0] == self.full_features.shape[0]
        self.size = self.onset_features.shape[0]
        self.weight = weight
        self.onset_ref = onset_ref

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        onset_features = self.onset_features[idx]
        if self.onset_ref is not None:
            onset_features = self.onset_features[idx] - self.onset_ref

        if self.weight is None:
            return onset_features, self.full_features[idx]

        return onset_features, self.full_features[idx], self.weight


class OnsetFeatureDataModule(L.LightningDataModule):
    """
    A LightningDataModule for datasets with onset features
    """

    def __init__(
        self,
        audio_path: Union[Path, str],  # Path to an audio file or directory
        feature: torch.nn.Module,  # A feature extractor
        onset_feature: torch.nn.Module,  # A feature extractor for short onsets
        sample_rate: int = 48000,  # Sample rate to compute features at
        batch_size: int = 64,  # Batch size
        return_norm: bool = False,  # Whether to return the feature norm
        center_onset: bool = False,  # Whether to return reference onset features as 0
        val_split: float = 0.0,  # Fraction of data to use for validation
        test_split: float = 0.0,  # Fraction of data to use from testing
        data_seed: int = 0,  # Seed for random data splits
    ):
        super().__init__()
        self.audio_path = Path(audio_path)
        self.feature = feature
        self.onset_feature = onset_feature
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.return_norm = return_norm
        self.center_onset = center_onset
        self.val_split = val_split
        self.test_split = test_split
        self.data_seed = data_seed

    def prepare_data(self) -> None:
        """
        Load the audio file and prepare the dataset
        """
        # Calculate the onset frames
        onset_frames = OnsetFrames(
            self.sample_rate,
            frame_size=self.sample_rate,
            on_thresh=10.0,
            wait=10000,
            backtrack=16,
            overlap_buffer=1024,
        )

        # Load audio files from a directory
        if self.audio_path.is_dir():
            audio_files = list(self.audio_path.glob("*.wav"))
            assert len(audio_files) > 0, "No audio files found in directory"
            log.info(f"Found {len(audio_files)} audio files.")

            audio = []
            for f in audio_files:
                x, sr = torchaudio.load(f)

                # Resample if necessary
                if sr != self.sample_rate:
                    x = torchaudio.transforms.Resample(sr, self.sample_rate)(x)

                # Onset detection and frame extraction to ensure all audio
                # is the same length and is aligned at an onset
                frames = onset_frames(x)
                frames = torch.from_numpy(frames).float()
                audio.append(frames)

            audio = torch.cat(audio, dim=0)
            log.info(f"{len(audio)} samples after onset detection.")

        elif self.audio_path.is_file and self.audio_path.suffix == ".wav":
            x, sr = torchaudio.load(self.audio_path)

            # Resample if necessary
            if sr != self.sample_rate:
                x = torchaudio.transforms.Resample(sr, self.sample_rate)(x)

            # Onset detection and frame extraction
            audio = onset_frames(x)
            audio = torch.from_numpy(audio).float()

            log.info(f"Found {len(audio)} samples.")

        else:
            raise RuntimeError("Invalid audio path")

        # Cache audio
        self.audio = audio

        # Compute full features
        self.full_features = self.feature(audio)
        loudsort = torch.argsort(self.full_features[:, 0], descending=True)
        idx = int(len(loudsort) * 0.5)
        idx = loudsort[idx]

        # Cache the index of the reference sample
        self.ref_idx = idx

        # Compute the difference between the features of each audio and the centroid
        self.diff = self.full_features - self.full_features[idx]
        assert torch.allclose(self.diff[idx], torch.zeros_like(self.diff[idx]))

        # Create a per feature weighting
        self.norm = torch.max(self.diff, dim=0)[0] - torch.min(self.diff, dim=0)[0]
        self.norm = torch.abs(1.0 / self.norm).float()

        # Compute onset features for each sample
        self.onset_features = self.onset_feature(audio)

        # Normalize onset features so each feature is in the range [0, 1]
        self.onset_features = self.onset_features - self.onset_features.min(dim=0)[0]
        self.onset_features = self.onset_features / self.onset_features.max(dim=0)[0]
        assert torch.all(self.onset_features >= 0.0)
        assert torch.all(self.onset_features <= 1.0)

        # Split the training data into train and test sets
        if self.test_split > 0.0:
            self.train_ids, self.test_ids = self.split_data(
                loudsort, self.ref_idx, self.test_split
            )
        else:
            self.train_ids = loudsort

        # Split the remaing data into train and validation sets
        if self.val_split > 0.0:
            self.train_ids, self.val_ids = self.split_data(
                self.train_ids, self.ref_idx, self.val_split
            )

        # Log the number of samples in each set
        log.info(f"Training samples: {len(self.train_ids)}")
        if hasattr(self, "val_ids"):
            log.info(f"Validation samples: {len(self.val_ids)}")
        if hasattr(self, "test_ids"):
            log.info(f"Test samples: {len(self.test_ids)}")

    def split_data(self, ids: torch.Tensor, ref_idx: int, split: float):
        """
        Select a subset of the data for validation
        """
        assert split > 0.0 and split < 1.0

        # Chunk the data into number of groups equal to the numbe of validation samples
        # and then select a random sample from each chunk.
        chunk_size = int(len(ids) * split) + 1
        chunks = torch.chunk(ids, chunk_size)

        train_ids = []
        val_ids = []

        g = torch.Generator()
        g.manual_seed(self.data_seed)
        for chunk in chunks:
            idx = torch.randint(0, len(chunk), (1,), generator=g).item()
            # Ensure the validation sample is not the reference sample
            if chunk[idx] == ref_idx:
                idx = (idx + 1) % len(chunk)

            val_ids.append(chunk[idx].item())
            train_ids.extend(chunk[chunk != chunk[idx]].tolist())

        assert len(train_ids) + len(val_ids) == len(ids)
        assert len(set(train_ids).intersection(set(val_ids))) == 0
        return torch.tensor(train_ids), torch.tensor(val_ids)

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """
        assert hasattr(self, "onset_features"), "Must call prepare_data() first"
        assert hasattr(self, "full_features"), "Must call prepare_data() first"

        onset_feature_ref = None
        if self.center_onset:
            onset_feature_ref = self.onset_features[self.ref_idx]

        norm = self.norm if self.return_norm else None
        if stage == "fit":
            self.train_dataset = OnsetFeatureDataset(
                self.onset_features[self.train_ids],
                self.diff[self.train_ids],
                norm,
                onset_ref=onset_feature_ref,
            )
            if hasattr(self, "val_ids"):
                self.val_dataset = OnsetFeatureDataset(
                    self.onset_features[self.val_ids],
                    self.diff[self.val_ids],
                    norm,
                    onset_ref=onset_feature_ref,
                )
        elif stage == "validate":
            if hasattr(self, "val_ids"):
                self.val_dataset = OnsetFeatureDataset(
                    self.onset_features[self.val_ids],
                    self.diff[self.val_ids],
                    norm,
                    onset_ref=onset_feature_ref,
                )
            else:
                raise ValueError("No validation data available")
        elif stage == "test":
            if hasattr(self, "test_ids"):
                self.test_dataset = OnsetFeatureDataset(
                    self.onset_features[self.test_ids],
                    self.diff[self.test_ids],
                    norm,
                    onset_ref=onset_feature_ref,
                )
            else:
                self.train_dataset = OnsetFeatureDataset(
                    self.onset_features,
                    self.diff,
                    norm,
                    onset_ref=onset_feature_ref,
                )
        else:
            raise NotImplementedError("Unknown stage")

    def train_dataloader(self, shuffle=True):
        batch_size = min(self.batch_size, len(self.train_dataset))
        if batch_size < self.batch_size:
            log.warning(
                f"Reducing batch size to {batch_size}, "
                "only that many samples available"
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            log.info("No test dataset available, using full dataset for testing")
            return self.train_dataloader(shuffle=False)

        log.info("Testing on the test dataset")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
