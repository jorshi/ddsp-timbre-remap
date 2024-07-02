import pytest
import torch

from timbreremap.loss import FeatureDifferenceLoss
from timbreremap.model import MLP
from timbreremap.synth import SimpleDrumSynth
from timbreremap.tasks import TimbreRemappingTask


@pytest.fixture
def features():
    # Dummy feature extractor
    return torch.nn.Linear(44100, 3)


@pytest.fixture
def preset_json(tmp_path):
    # Crate a dummy preset json
    synth = SimpleDrumSynth(sample_rate=44100, num_samples=44100)
    num_params = synth.get_num_params()
    preset = torch.rand(1, num_params)
    preset_dict = synth.get_param_dict()
    for i, (k, n) in enumerate(synth.get_param_dict().items()):
        preset_dict[k] = n.from_0to1(preset[0, i]).item()
    preset_json = tmp_path.joinpath("preset.json")
    synth.save_params_json(preset_json, preset_dict)
    return preset_json


def test_init_timbre_remapping_task(features, preset_json):
    synth = SimpleDrumSynth(sample_rate=44100, num_samples=44100)
    model = MLP(in_size=1, hidden_size=1, out_size=1, num_layers=1)
    _ = TimbreRemappingTask(
        model=model,
        synth=synth,
        preset=preset_json,
        feature=features,
        loss_fn=None,
    )


def test_timbre_remapping_task_forward(features, preset_json):
    synth = SimpleDrumSynth(sample_rate=44100, num_samples=44100)
    num_params = synth.get_num_params()
    model = MLP(in_size=1, hidden_size=1, out_size=num_params, num_layers=1)
    mapping = TimbreRemappingTask(
        model=model,
        synth=synth,
        preset=preset_json,
        feature=features,
        loss_fn=None,
    )

    inputs = torch.rand(1, 1)
    y = mapping(inputs)
    assert y.shape == (1, 44100)


def test_timbre_remapping_task_train_step(features, preset_json):
    # Initialize a synth and preset
    synth = SimpleDrumSynth(sample_rate=44100, num_samples=44100)
    num_params = synth.get_num_params()

    # Initialize a parameter mapping model
    mapping = TimbreRemappingTask(
        model=torch.nn.Linear(1, num_params),
        synth=synth,
        preset=preset_json,
        feature=features,
        loss_fn=FeatureDifferenceLoss(),
    )

    # Create a dummy batch of inputs
    inputs = torch.rand(1, 1)
    target = torch.rand(1, 1)
    loss = mapping.training_step((inputs, target), 0)

    # Check that the loss is a scalar
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
