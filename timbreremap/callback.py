"""
PyTorch Lightning Callbacks
"""
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
from einops import rearrange
from einops import repeat

from timbreremap.data import OnsetFeatureDataset
from timbreremap.export import ParameterMapper


class SaveAudioCallback(L.Callback):
    def __init__(self, num_samples: int = 16):
        super().__init__()
        self.num_samples = num_samples

    def on_train_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        data = trainer.train_dataloader.dataset

        # If the onset reference values are being used in an OnsetFeatureDataset,
        # pass those values in to normalize the input features
        onset_ref = None
        if isinstance(data, OnsetFeatureDataset):
            onset_ref = data.onset_ref

        self.render_audio(module, onset_ref)
        self.render_fig(module, onset_ref)

    def render_audio(self, module, onset_ref=None):
        outdir = Path(module.logger.log_dir).joinpath("audio")
        outdir.mkdir(exist_ok=True)

        num_features = module.model.in_size

        # Generate input ranging from 0 to 1
        x = torch.linspace(0, 1, self.num_samples, device=module.device)
        x = repeat(x, "n -> n f", f=num_features)

        # Offset onset features
        if onset_ref is not None:
            x = x - onset_ref.to(module.device)

        # Generate audio
        y = module(x)
        y = rearrange(y, "b n -> 1 (b n)")
        y = y.detach().cpu()

        torchaudio.save(
            outdir.joinpath("gradient_all.wav"), y, module.synth.sample_rate
        )

    def render_fig(self, module, onset_ref=None):
        # Save a plot of parameter changes
        outdir = Path(module.logger.log_dir).joinpath("plots")
        outdir.mkdir(exist_ok=True)

        # Generate input ranging from 0 to 1
        x = torch.linspace(0, 1, self.num_samples * 100, device=module.device)
        x = repeat(x, "n -> n f", f=module.model.in_size)

        # Offset onset features
        if onset_ref is not None:
            print(onset_ref)
            x = x - onset_ref.to(module.device)

        labels = list(module.synth.get_param_dict().keys())

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        preset = module.preset.detach().cpu().numpy()
        param_mod = module.model(x).detach().cpu().numpy()
        damping = module.damping.detach().cpu().numpy()

        for i in range(preset.shape[-1]):
            mod = param_mod[:, i] * damping[0, i]
            ax.plot(mod, label="-".join(labels[i]))

        fig.legend(loc=7)
        fig.subplots_adjust(right=0.75)
        fig.savefig(outdir.joinpath("parameter_modulationss.png"), dpi=150)


class SaveTorchScriptCallback(L.Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        model = module.model.cpu()
        damping = module.damping.detach().cpu()
        preset = module.preset.detach().cpu()
        mapper = ParameterMapper(model, damping, preset)
        sm = torch.jit.script(mapper)

        # Test the torchscript module
        x = torch.rand(1, module.model.in_size)
        y = sm(x)

        num_params = module.synth.get_num_params()
        assert y.shape == (2, num_params)

        # Save the torchscript module
        outdir = Path(module.logger.log_dir).joinpath("torchscript")
        outdir.mkdir(exist_ok=True)
        torch.jit.save(sm, outdir.joinpath("drum_mapper.pt"))


class SaveTimbreIntervalAudio(L.Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        outdir = Path(module.logger.log_dir).joinpath("audio")
        outdir.mkdir(exist_ok=True)

        # Save the reference and target audio
        data = trainer.train_dataloader.dataset
        torchaudio.save(
            outdir.joinpath("A.wav"), data.reference_audio, data.sample_rate
        )
        torchaudio.save(outdir.joinpath("B.wav"), data.target_audio, data.sample_rate)

        # Generate audio for the timbre intervals
        y_true = module.synth(module.preset).detach().cpu()
        y_pred = module().detach().cpu()

        torchaudio.save(outdir.joinpath("C.wav"), y_true, module.synth.sample_rate)
        torchaudio.save(outdir.joinpath("D.wav"), y_pred, module.synth.sample_rate)

        # What is the error per feature?
        ref_true = module.feature(y_true)
        ref_pred = module.feature(y_pred)

        # Calculate feature differences
        diff = ref_pred - ref_true
        error = diff - data[0]
        error = error[0].tolist()

        meta = []
        for feature in module.feature.features:
            frame_size = feature.frame_size
            hop_size = feature.hop_size
            for f in feature.flattened_features:
                meta.append(
                    {
                        "feature": f"{f[0]}.{f[1]}",
                        "frame_size": frame_size,
                        "hop_size": hop_size,
                    }
                )

        feature_error = []
        for e, f in zip(error, meta):
            feature_error.append({**f, "error": e})

        df = pd.DataFrame(feature_error)
        outdir = Path(module.logger.log_dir)
        df.to_csv(outdir.joinpath("feature_error.csv"), index=False)
