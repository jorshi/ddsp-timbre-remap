"""
timbre remapping cli entry point
"""
import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.cli import LightningCLI
from tqdm import tqdm


# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def run_cli():
    """ """
    _ = LightningCLI()
    return


def main():
    """ """
    start_time = time.time()
    run_cli()
    end_time = time.time()
    log.info(f"Total time: {end_time - start_time} seconds")


def test_version():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("train_logs", help="Input log directory", type=str)
    parser.add_argument("output_dir", help="Output log directory", type=str)

    # Get input directory
    args = parser.parse_args(sys.argv[1:])
    logs = Path(args.train_logs)
    output = Path(args.output_dir)

    # Reset arguments
    args = sys.argv[:1]
    configs = []
    for version in sorted(logs.iterdir()):
        if version.is_dir():
            config = version.joinpath("config.yaml")
            ckpt = list(version.joinpath("checkpoints").glob("*.ckpt"))
            if len(ckpt) > 0:
                ckpt = ckpt[0]
            else:
                continue

            if config.exists():
                cfg_args = ["test", "-c", str(config), "--ckpt", str(ckpt)]
                configs.append(cfg_args)

    for c in configs:
        run_args = copy.deepcopy(args)
        run_args.extend(c)
        run_args.extend(["--trainer.logger", "CSVLogger"])
        run_args.extend(["--trainer.logger.save_dir", str(output)])
        sys.argv = run_args
        print(run_args)
        _ = LightningCLI()

    return


def train_sdss():
    """ """
    args = sys.argv
    configs = []
    folders = Path("audio/sdss_filtered")
    for f in folders.iterdir():
        if f.is_dir():
            cfg_args = copy.deepcopy(args)
            cfg_args.extend(["--data.audio_path", str(f)])
            configs.append(cfg_args)

    for i, c in enumerate(configs):
        sys.argv = c
        print(c)
        _ = LightningCLI()

    return


def optimize_sdss():
    """
    Optimize the synthesis parameters for the sdss dataset
    """
    args = sys.argv
    configs = []
    folders = Path("audio/sdss_filtered")
    for f in folders.iterdir():
        if f.is_dir():
            cfg_args = copy.deepcopy(args)
            cfg_args.extend(["--data.audio_path", str(f)])
            configs.append(cfg_args)

    outdir = Path("experiment/test_logs_direct_opt").joinpath("lightning_logs")
    outdir.mkdir(exist_ok=True, parents=True)

    # Get the starting version number from the output directory
    versions = sorted(outdir.glob("version_*"))
    version = len(versions)
    log.info(f"Starting version: {version}")

    # Run the optimization for each configuration
    for i, c in enumerate(configs):
        sys.argv = c
        print(c)

        # Create the output directory
        outdir_i = outdir.joinpath(f"version_{version + i}")
        outdir_i.mkdir(exist_ok=True)
        output = outdir_i.joinpath("metrics.csv")

        # Run the optimization
        direct_optimization(output=output)

    return


def direct_optimization(output=None):
    """
    Direct optimization of synthesis parameters
    """
    cli = LightningCLI(run=False)

    # Set up the training dataset with no validation split
    datamodule = cli.datamodule
    datamodule.batch_size = 256
    datamodule.prepare_data()
    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()

    for i, item in enumerate(dataloader):
        _optimize_synth(cli.model, target=item[1])

    metrics = {}
    for i, (k, v) in enumerate(cli.model.feature_metrics.items()):
        metrics[f"test/{k}"] = [
            v.compute().item(),
        ]

    df = pd.DataFrame.from_dict(metrics, orient="columns")
    if output is not None:
        df.to_csv(output, index=False)
    else:
        df.to_csv("metrics.csv", index=False)

    return


def _optimize_synth(model: L.LightningModule, target: torch.Tensor):
    """
    Optimize the synthesis parameters to match a feature target
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    target = target.to(device)

    synth = model.synth
    preset = model.preset
    modulation = torch.zeros(
        target.shape[0], preset.shape[1], device=device, requires_grad=True
    )

    reference = model.feature(synth(preset))

    # Create an optimizer
    optimizer = torch.optim.Adam([modulation], lr=0.005)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5, verbose=True
    )

    pbar = tqdm(range(1000))
    for i in pbar:
        optimizer.zero_grad()

        params = preset + modulation
        params = torch.clip(params, 0.0, 1.0)

        y = synth(params)
        y_features = model.feature(y)

        loss = model.loss_fn(y_features, reference, target)
        loss.backward()
        optimizer.step()

        schedule.step(loss)

        pbar.set_description(f"Loss: {loss.detach().item():.4f}")

    diff = y_features - reference
    for i, (k, v) in enumerate(model.feature_metrics.items()):
        v.update(diff[..., i], target[..., i])
