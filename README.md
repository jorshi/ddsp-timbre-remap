<div align="center">

# Real-time Timbre Remapping with Differentiable DSP


[![Demo](https://img.shields.io/badge/Web-Audio_Examples-blue)](https://jordieshier.com/projects/nime2024/)
[![Paper](https://img.shields.io/badge/PDF-Paper-green)](http://instrumentslab.org/data/andrew/shier_nime2024.pdf)

[Jordie Shier](https://jordieshier.com), [Charalampos Saitis](http://eecs.qmul.ac.uk/people/profiles/saitischaralampos.html), Andrew Robertson, and [Andrew McPherson](https://www.imperial.ac.uk/people/andrew.mcpherson)

</div>

This repository contains training code for our *Real-time Timbre Remapping with Differentiable DSP*, presented at NIME 2024.
This research explores the application of differentiable digital signal processing (DDSP) towards
timbral control of an audio synthesizer. 

Audio features are extracted from input percussive audio (i.e., drums) at detected onsets and are mapped
to synthesizer parameter modulations using neural networks. There neural networks are trained to estimated parameter modulations that result in desired timbral and dynamic modulations for a differentiable drum synth based on the Roland TR-808.

**Coming Soon** -- export trained models and load them in an audio plug-in for real-time timbre remapping.


## Install
Clone the repo and then install the `timbreremap` package. Requires Python version 3.9 or greater.

```bash
pip install --upgrade pip
pip install -e .
```

## Examples

**Coming soon**


## Numerical Experiments

Instructions to reproduce numerical results from the NIME 2024 paper. The included
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
