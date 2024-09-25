# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: remap
#     language: python
#     name: python3
# ---
# +
import argparse
from pathlib import Path

import IPython.display as ipd
import matplotlib.pyplot as plt
import torch
import torchaudio

import timbreremap.feature as feature
from timbreremap.cli import _optimize_synth
from timbreremap.data import OnsetFeatureDataModule
from timbreremap.loss import FeatureDifferenceLoss
from timbreremap.np import OnsetFrames
from timbreremap.synth import Snare808
from timbreremap.tasks import TimbreRemappingTask

# %load_ext autoreload
# %autoreload 2

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--data_path", type=str, default="audio/carson_gant_drums/performance.wav"
)
argparser.add_argument("--iters", type=int, default=100)

args = argparser.parse_args()


# +
sample_rate = 48000
data_path = args.data_path
# data_path = 'audio/percussion_1.wav'

N_ITERS = args.iters
# -

drums, sr = torchaudio.load(data_path)
drums = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(drums)[:1]

onset_extractor = OnsetFrames(sr, 256)
onset_times = onset_extractor.onset(drums)

# +
loudness_extractor = feature.Loudness(sample_rate=sample_rate)
sc_extractor = feature.SpectralCentroid(
    sample_rate=sample_rate, scaling="kazazis", floor=1e-4
)
sf_extractor = feature.SpectralFlatness()
tc_extractor = feature.TemporalCentroid(sample_rate=sample_rate, scaling="schlauch")

frame_extractor = feature.CascadingFrameExtactor(
    [loudness_extractor, sc_extractor, sf_extractor],
    [
        1,
        4,
    ],
    2048,
    512,
)
global_extractor = feature.CascadingFrameExtactor(
    [tc_extractor],
    [
        1,
    ],
    5512,
    5512,
)
extractor = feature.FeatureCollection([frame_extractor, global_extractor])

onset_extractor = feature.CascadingFrameExtactor(
    [loudness_extractor],
    [
        1,
    ],
    256,
    256,
)

# +
data = OnsetFeatureDataModule(
    data_path, extractor, onset_extractor, sample_rate, return_norm=True
)
data.prepare_data(ref_idx=4)
data.setup("fit")
dataset = data.train_dataset

print(data.ref_idx)
# -

idx_map = list(data.loudsort.numpy())
print(idx_map)

# +
feats = []
for i, audio in enumerate(data.audio):
    idx = idx_map.index(i)
    feat = dataset[idx][1][None, ...]
    feats.append(feat)
    ipd.display(ipd.Audio(audio.squeeze().numpy(), rate=sample_rate))

feats = torch.cat(feats, dim=0)

# +
labels = [f"{k[0]}:{k[1]}" for f in extractor.features for k in f.flattened_features]
print(len(labels))

# create subplots for each feature
# fig, axs = plt.subplots(feats.shape[-1], 1, figsize=(10, 3 * feats.shape[-1]))
# for i in range(feats.shape[-1]):
#     axs[i].plot(feats[:, i].numpy())
#     axs[i].set_title(labels[i])

# plt.tight_layout()

# +
synth = Snare808(
    sample_rate=sample_rate,
    num_samples=sample_rate,
    buffer_noise=True,
    buffer_size=sample_rate,
)

preset = "808_snare_1.json"
preset = f"../cfg/presets/{preset}"

parameters, _ = synth.load_params_json(preset)
audio = synth(parameters)
ipd.Audio(audio, rate=sample_rate)
# -

# # Create a set of modulated presets

model = torch.nn.Identity()
task = TimbreRemappingTask(
    model=model,
    synth=synth,
    feature=extractor,
    loss_fn=FeatureDifferenceLoss(),
    preset=preset,
)


# +
def run_optimization(task, target, ref_idx=None, iterations=100, target_scale=1.0):
    data.prepare_data(ref_idx=ref_idx)
    data.setup("fit")
    dataset = data.train_dataset

    modulations = []
    target_features = []
    for d in dataset:
        target = d[1][None, ...]
        target = target * target_scale
        target_features.append(target)
        modulation = _optimize_synth(task, target, iterations=iterations, norm=d[2])
        modulations.append(modulation)

    return modulations, torch.cat(target_features, dim=0)


modulations, target_features = run_optimization(task, feats, ref_idx=0, iterations=10)


# -


def resynthesize(mods, synth):
    # Resynthesize the audio with the modulations
    audio = []
    synth = synth.to("cpu")

    # Get synth features
    synth_audio = synth(parameters)
    ref_features = extractor(synth_audio)
    synth_features = []
    idx_order = []

    for i in range(len(mods)):
        idx = idx_map.index(i)
        params = parameters + mods[idx].detach().cpu()
        params = torch.clip(params, 0.0, 1.0)
        audio.append(synth(params))

        pred_features = extractor(audio[-1])
        diff = pred_features - ref_features
        synth_features.append(diff)
        idx_order.append(idx)

    audio = torch.cat(audio, dim=0)

    # Stitch back together with the onsets
    resynth = torch.zeros_like(drums)
    for i, onset in enumerate(onset_times):
        start = onset
        end = min(onset + audio[i].shape[-1], resynth.shape[-1])
        resynth[0, start:end] += audio[i][: end - start]

    return resynth, torch.cat(synth_features, dim=0), idx_order


# +
resynth, synth_features, idx_order = resynthesize(modulations, synth)

ipd.Audio(resynth.detach().cpu().numpy(), rate=sr)


# +
def plot_features(target_features, synth_features, idx_order, extractor):
    # create subplots for each feature
    labels = [
        f"{k[0]}:{k[1]}" for f in extractor.features for k in f.flattened_features
    ]
    fig, axs = plt.subplots(
        target_features.shape[-1], 1, figsize=(10, 3 * feats.shape[-1])
    )
    for i in range(target_features.shape[-1]):
        axs[i].plot(target_features[idx_order, i].numpy(), label="target")
        axs[i].plot(synth_features[:, i].numpy(), label="synth")
        axs[i].set_title(labels[i])

    plt.tight_layout()
    plt.legend()
    return fig


fig = plot_features(target_features, synth_features, idx_order, extractor)
# -

audios = []
for i in range(len(dataset)):
    print(f"Optimizing for {i}")
    modulations, y_feat = run_optimization(task, feats, ref_idx=i, iterations=N_ITERS)
    resynth, y_hat_faat, idx = resynthesize(modulations, synth)
    audios.append(resynth.detach().cpu())

    output = Path(f"results/anchor_swap_{Path(data_path).stem}_{i}.png")
    fig = plot_features(y_feat, y_hat_faat, idx, extractor)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    torchaudio.save(output.with_suffix(".wav"), resynth.detach().cpu(), sample_rate)

# +
audio_tensor = torch.hstack(audios[0:])

output = Path(f"results/anchor_swap_{Path(data_path).stem}_concat.wav")
torchaudio.save(output, audio_tensor, sample_rate)

ipd.display(ipd.Audio(audio_tensor.squeeze().numpy(), rate=sample_rate))
ipd.display(ipd.Audio(drums.squeeze().numpy(), rate=sample_rate))
# -

scale = torch.hstack([torch.linspace(0.0, 1.0, 4), torch.linspace(1.0, 10.0, 4)])
print(scale)

# +
audios = []

for i in range(len(scale)):
    print(f"Optimizing for {i}")
    modulations, y_feat = run_optimization(
        task, feats, ref_idx=0, iterations=N_ITERS, target_scale=scale[i]
    )
    resynth, y_hat_faat, idx = resynthesize(modulations, synth)
    audios.append(resynth.detach().cpu())

    output = Path(f"results/target_scale_{Path(data_path).stem}_{i}.png")
    fig = plot_features(y_feat, y_hat_faat, idx, extractor)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    torchaudio.save(output.with_suffix(".wav"), resynth.detach().cpu(), sample_rate)

# +
audio_tensor = torch.hstack(audios[0:])

output = Path(f"results/target_scale_{Path(data_path).stem}_concat.wav")
torchaudio.save(output, audio_tensor, sample_rate)

ipd.display(ipd.Audio(audio_tensor.squeeze().numpy(), rate=sample_rate))
ipd.display(ipd.Audio(drums.squeeze().numpy(), rate=sample_rate))
