"""
PyTorch Lightning modules for training models
"""
import logging
import os

import lightning as L
import torch
from einops import repeat
from torchmetrics import Metric

from timbreremap.feature import CascadingFrameExtactor
from timbreremap.feature import FeatureCollection
from timbreremap.synth import AbstractSynth

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class TimbreRemappingTask(L.LightningModule):
    """
    A LightningModule to train a synthesizer timbre remapping model
    """

    def __init__(
        self,
        model: torch.nn.Module,  # Parameter mapping model
        synth: AbstractSynth,  # Synthesizer
        feature: torch.nn.Module,  # A feature extractor
        loss_fn: torch.nn.Module,  # A loss function
        preset: str = None,  # The preset to be modulated (loaded from a json)
    ):
        super().__init__()
        self.model = model
        self.synth = synth
        self.feature = feature
        self.loss_fn = loss_fn

        preset, damping = load_preset_and_damping(synth, preset)
        self.register_buffer("preset", preset)
        log.info(f"Loaded preset: {preset}")

        if damping is not None:
            self.register_buffer("damping", damping)
            log.info(f"Loaded damping: {damping}")

        # Compute the reference features from the preset
        reference = self.feature(self.synth(self.preset))
        self.register_buffer("reference", reference)

        # Setup feature metrics if the feature is a CascadingFrameExtactor
        feature_metrics = {}
        pretrain_feature_metrics = {}
        labels = []
        if isinstance(self.feature, CascadingFrameExtactor):
            labels.extend(self.feature.flattened_features)
        elif isinstance(self.feature, FeatureCollection):
            for feature in self.feature.features:
                if isinstance(feature, CascadingFrameExtactor):
                    labels.extend(feature.flattened_features)
        else:
            log.warning(
                "Feature is not a CascadingFrameExtactor, "
                "feature metrics will not be calculated"
            )
        for label in labels:
            feature_metrics["_".join(label)] = FeatureErrorMetric()
            pretrain_feature_metrics["pre" + "_".join(label)] = FeatureErrorMetric()
        self.feature_metrics = torch.nn.ModuleDict(feature_metrics)
        self.pretrain_feature_metrics = torch.nn.ModuleDict(pretrain_feature_metrics)

    def forward(self, inputs: torch.Tensor):
        # Pass the input features through a the parameter mapping model
        param_mod = self.model(inputs)

        # Apply parameter-wise damping if provided
        if hasattr(self, "damping"):
            param_mod = param_mod * self.damping

        # Modulate the preset with output of the parameter mapping model
        assert param_mod.shape[-1] == self.preset.shape[-1]
        params = self.preset + param_mod
        params = torch.clip(params, 0.0, 1.0)

        # Pass the output of the parameter mapping model through the synth
        y = self.synth(params)
        return y

    def _do_step(self, batch, batch_idx):
        if len(batch) == 3:
            inputs, target, norm = batch
        else:
            inputs, target = batch
            norm = 1.0

        y = self(inputs)

        # Calculate features from the input audio
        y_features = self.feature(y)

        # Calculate the feature difference loss
        # TODO: add feature norm? Should it come from the dataset?
        # Can this actually just be something like LayerNorm in the loss module?
        features = (y_features, self.reference, target, norm)
        loss = self.loss_fn(y_features, self.reference, target, norm)
        return loss, features, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._do_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._do_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, features, _ = self._do_step(batch, batch_idx)
        self.log("test/loss", loss, prog_bar=True)

        y_features, reference, target, norm = features
        diff = y_features - reference

        for i, (k, v) in enumerate(self.feature_metrics.items()):
            v.update(diff[..., i], target[..., i])
            self.log(f"test/{k}", v, prog_bar=True)

        for i, (k, v) in enumerate(self.pretrain_feature_metrics.items()):
            ref = repeat(reference, "1 n -> b n", b=target.shape[0])
            v.update(ref[..., i], target[..., i])
            self.log(f"test/{k}", v, prog_bar=True)

        return loss


def load_preset_and_damping(synth: AbstractSynth, preset: str = None):
    # Try loading the preset
    damping = None
    if preset is not None:
        preset, damping = synth.load_params_json(preset)
    else:
        log.warning("No preset provided, using random preset")
        preset = torch.rand(1, synth.get_num_params())

    if preset.shape != (1, synth.get_num_params()):
        raise ValueError(
            f"preset must be of shape (1, {synth.get_num_params()}), "
            f"received {preset.shape}"
        )

    # Register the damping if provided
    if damping is not None:
        if damping.shape != (1, synth.get_num_params()):
            raise ValueError(
                f"damping must be of shape (1, {synth.get_num_params()}), "
                f"received {damping.shape}"
            )

    return preset, damping


class FeatureErrorMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, diff: torch.Tensor, target: torch.Tensor) -> None:
        assert diff.shape == target.shape
        error = torch.abs(diff - target)
        self.error += torch.sum(error)
        self.total += diff.shape[0]

    def compute(self) -> torch.Tensor:
        return self.error.float() / self.total
