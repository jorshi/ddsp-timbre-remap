import torch

from timbreremap.synth import CrossFade
from timbreremap.synth import ExponentialDecay
from timbreremap.synth import ParamaterNormalizer


def test_parameter_normalizer_init():
    normalizer = ParamaterNormalizer(0.0, 1.0)
    assert normalizer.min_value == 0.0
    assert normalizer.max_value == 1.0


def test_parameter_normalizer_from_0to1():
    normalizer = ParamaterNormalizer(10.0, 20.0)
    x = torch.linspace(0.0, 1.0, 10)
    y = normalizer.from_0to1(x)
    expected = torch.linspace(10.0, 20.0, 10)
    assert torch.allclose(y, expected)


def test_parameter_normalizer_from_1to0():
    normalizer = ParamaterNormalizer(10.0, 20.0)
    x = torch.linspace(10.0, 20.0, 10)
    y = normalizer.to_0to1(x)
    expected = torch.linspace(0.0, 1.0, 10)
    assert torch.allclose(y, expected)


def test_exponential_decay_init():
    decay = ExponentialDecay(sample_rate=44100)
    assert decay.normalizers["decay"].min_value == 10.0


def test_exponential_decay_forward():
    env = ExponentialDecay(sample_rate=44100)
    decay = torch.tensor([0.1, 0.5])[..., None]
    y = env(1000, decay)
    assert y.shape == (2, 1000)


def test_crossfade_init():
    fade = CrossFade(sample_rate=44100)
    assert fade.normalizers["fade"].min_value == 0.0
    assert fade.normalizers["fade"].max_value == 1.0
