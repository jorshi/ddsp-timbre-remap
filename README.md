<div align="center">

# Real-time Timbre Remapping with Differentiable DSP


[![Demo](https://img.shields.io/badge/Web-Audio_Examples-blue)](https://jordieshier.com/projects/nime2024/)
[![Paper](https://img.shields.io/badge/PDF-Paper-green)](http://instrumentslab.org/data/andrew/shier_nime2024.pdf)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nwV5y2eYiCF9YIM1BSmKU9uiPBbw9OxV?usp=sharing)

[Jordie Shier](https://jordieshier.com), [Charalampos Saitis](http://eecs.qmul.ac.uk/people/profiles/saitischaralampos.html), Andrew Robertson, and [Andrew McPherson](https://www.imperial.ac.uk/people/andrew.mcpherson)

</div>

This repository contains training code for our paper *Real-time Timbre Remapping with Differentiable DSP*,
which explored the application of differentiable digital signal processing (DDSP) for
audio-driven control of a synthesizer.

Audio features extracted from input percussive audio (i.e., drums) at detected onsets are mapped
to synthesizer parameter modulations using neural networks. These neural networks are trained to estimate parameter modulations to create desired timbral and dynamic changes on a synthesizer. We developed
a differentiable drum synthesizer based on the Roland TR-808, which enables optimization and
estimation of synthesizer parameters based on audio features extracted from synthesized audio.

For more details on the research check out the [paper](http://instrumentslab.org/data/andrew/shier_nime2024.pdf) and listen to examples on the [research webpage](https://jordieshier.com/projects/nime2024/).

[TorchDrum](https://github.com/jorshi/torchdrum-plugin) -- export trained models into an
audio plug-in we developed alongside this research for real-time timbre remapping in your DAW.

## Install
Clone the repo and then install the `timbreremap` package. Requires Python version 3.9 or greater.

```bash
pip install --upgrade pip
pip install -e .
```

## Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nwV5y2eYiCF9YIM1BSmKU9uiPBbw9OxV?usp=sharing)

**Training a model**

Example of training a new mapping model on the cpu for 50 epochs. Replace `audio/nime-demo/break.wav` with your own audio file (or folder of audio files). Results will be saved in a directoy called `lightning_logs` ordered by versions, which will include a compiled mapping model in the `torchscript` folder.
```bash
timbreremap fit -c cfg/onset_mapping_808.yaml --data.audio_path audio/nime-demo/break.wav --trainer.accelerator cpu --trainer.max_epochs 50
```

## Numerical Experiments

Instructions to reproduce numerical results from the [NIME 2024 paper](https://arxiv.org/abs/2407.04547). The included
scripts require a GPU to run.

### Dataset

Download the [Snare Drum Data Set (SDSS)](https://aes2.org/publications/elibrary-page/?id=20912).
We used a subset of this dataset, which can be downloaded as follows:

```bash
mkdir audio
cd audio
wget https://pub-814e66019388451395cf43c0b6f10300.r2.dev/sdss_filtered.zip
unzip sdss_filtered.zip
```

### Training

The following scripts will run a series of trainings iterating over the snare drum
dataset and five different synthesizer presets. In total, 240 models are trained for
each mapping algorithm.
Each model takes around 2min to train on a GPU, which means training all models will take around 24 hours.

```bash
./scripts/train_linear.sh && ./scripts/train_mlp.sh && ./scripts/train_mlp_lrg.sh
```

To run the baseline, which involves no neural network, just estimating the synthesis parameter directly using gradient descent:

```bash
./scripts/direct_optimize.sh
```

### Results
To compile results from the numerical experiments into a summary table:

```bash
python scripts/results.py experiment
```

Which will output a file named `table.tex` with a table similar to the one presented in the paper.
