"""
Unit tests for timbre remapping models
"""
import torch

from timbreremap.model import LinearMapping


def test_linear_mapping_init():
    model = LinearMapping(3, 8)
    assert isinstance(model, LinearMapping)
    assert len(list(model.children())) == 1
    assert model.net.bias is None
    assert model.net.weight.shape == (8, 3)


def test_linear_mapping_forward():
    model = LinearMapping(2, 3)
    torch.nn.init.constant_(model.net.weight, 1.0)
    y = model(torch.tensor([0.5, 1.0]))
    assert torch.all(y == 1.5)


def test_linear_mapping_clamp():
    model = LinearMapping(2, 3, clamp=True)
    torch.nn.init.constant_(model.net.weight, 0.5)
    y = model(torch.tensor([2.0, 4.0]))
    assert torch.all(y == 1.0)
    y = model(torch.tensor([2.0, -4.0]))
    assert torch.all(y == -1.0)
