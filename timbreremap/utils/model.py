"""
Helpful utils for handling pre-trained models
"""
import sys
from typing import List
from typing import Optional
from unittest.mock import patch

import lightning as L
import torch
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.cli import LightningCLI


class CustomCLI(LightningCLI):
    """
    PyTorch Lightning CLI
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument("--ckpt_path", type=str, help="Placeholder")


def load_model(
    config: str,
    ckpt: Optional[str] = None,
    device: str = "cpu",
    extra_args: Optional[List[str]] = None,
    load_data: bool = True,
):
    """
    Load a model from a checkpoint using a config file.
    """
    args = ["fit", "-c", str(config), "--trainer.accelerator", device]
    if extra_args is not None:
        args.extend(extra_args)

    datamodule = None
    if not load_data:
        datamodule = L.LightningDataModule

    with patch.object(sys, "argv", args):
        cli = CustomCLI(run=False, datamodule_class=datamodule)
        model = cli.model

    if ckpt is not None:
        state_dict = torch.load(ckpt, map_location=device)["state_dict"]
        model.load_state_dict(state_dict)

    return model, cli
