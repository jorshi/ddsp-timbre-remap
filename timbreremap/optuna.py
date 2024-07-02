"""
Hyperparameter optimization using Optuna CLI
"""
import copy
import logging
import os
import sys
from functools import partial

import optuna
from lightning.pytorch.cli import LightningCLI
from optuna.integration import PyTorchLightningPruningCallback

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def objective(trial, args=None):
    n_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_size = trial.suggest_categorical(
        "hidden_size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    )
    init_var = trial.suggest_float("init_var", 1e-7, 1.0, log=True)
    activation = trial.suggest_categorical(
        "activation",
        [
            "torch.nn.LeakyReLU",
            "torch.nn.ReLU",
            "torch.nn.Sigmoid",
            "torch.nn.Tanh",
        ],
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 0.01, log=True)
    scale_output = trial.suggest_categorical("scale_output", [True, False])

    cfg_args = copy.deepcopy(args)
    cfg_args.extend(["--trainer.max_epochs", "200"])

    # Hyperparameters
    cfg_args.extend(["--model.model.num_layers", str(n_layers)])
    cfg_args.extend(["--model.model.hidden_size", str(hidden_size)])
    cfg_args.extend(["--model.model.init_var", str(init_var)])
    cfg_args.extend(["--model.model.activation", str(activation)])
    cfg_args.extend(["--model.model.scale_output", str(scale_output)])

    cfg_args.extend(["--optimizer.lr", str(learning_rate)])

    sys.argv = cfg_args

    cli = LightningCLI(run=False)
    cli.trainer.callbacks.append(
        PyTorchLightningPruningCallback(trial, monitor="val/loss")
    )

    try:
        cli.trainer.fit(cli.model, cli.datamodule)
    except optuna.TrialPruned as e:
        log.info(e)
        return cli.trainer.callback_metrics["val/loss"].item()
    except Exception as e:
        log.error(e)
        return None

    assert cli.trainer.callback_metrics["val/loss"].item() is not None
    return cli.trainer.callback_metrics["val/loss"].item()


def objective_linear(trial, args=None):
    init_var = trial.suggest_float("init_std", 1e-7, 1.0, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 0.01, log=True)

    cfg_args = copy.deepcopy(args)
    cfg_args.extend(["--trainer.max_epochs", "200"])

    # Hyperparameters
    cfg_args.extend(["--model.model.init_std", str(init_var)])
    cfg_args.extend(["--optimizer.lr", str(learning_rate)])

    sys.argv = cfg_args

    cli = LightningCLI(run=False)
    cli.trainer.callbacks.append(
        PyTorchLightningPruningCallback(trial, monitor="val/loss")
    )

    try:
        cli.trainer.fit(cli.model, cli.datamodule)
    except optuna.TrialPruned as e:
        log.info(e)
        return cli.trainer.callback_metrics["val/loss"].item()
    except Exception as e:
        log.error(e)
        return None

    assert cli.trainer.callback_metrics["val/loss"].item() is not None
    return cli.trainer.callback_metrics["val/loss"].item()


def run_optuna():
    """ """
    args = sys.argv
    if "--linear" in args:
        args.remove("--linear")
        args.extend(["--model.model", "cfg/models/linear_mapper.yaml"])
        objective_func = partial(objective_linear, args=args)
        name = "Linear Mapper"
    else:
        objective_func = partial(objective, args=args)
        name = "MLP Mapper"

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///db.sqlite3",
        study_name=f"808 Snare HyperParam Jan29 - {name}",
        load_if_exists=True,
    )
    study.optimize(objective_func, n_trials=200)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return
