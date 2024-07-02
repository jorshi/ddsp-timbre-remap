import numpy as np
import torch

from timbreremap.feature import FeatureExtractor
from timbreremap.feature import OnsetSegment
from timbreremap.feature import RMS
from timbreremap.feature import SpectralCentroid
from timbreremap.feature import SpectralFlatness


def test_onset_segment():
    x = torch.arange(100).unsqueeze(0).float()
    seg = OnsetSegment(50)
    y = seg(x)
    assert y.shape == (1, 50)
    assert torch.all(y == torch.arange(50).unsqueeze(0).float())


def test_onset_segment_delay():
    x = torch.arange(100).unsqueeze(0).float()
    seg = OnsetSegment(50, 10)
    y = seg(x)
    assert y.shape == (1, 50)
    assert torch.all(y == torch.arange(10, 60).unsqueeze(0).float())


def test_rms():
    x = torch.ones(100).unsqueeze(0).float()
    rms = RMS()
    y = rms(x)
    assert y.shape == (1,)
    assert y.item() == 1.0


def test_rms_db():
    x = torch.ones(100).unsqueeze(0).float()
    rms = RMS(db=True)
    y = rms(x)
    assert y.shape == (1,)
    assert y.item() == 0.0


def test_spectral_centroid():
    sr = 44100
    f0 = sr / 128.0
    w0 = 2.0 * torch.pi * (f0 / sr)
    phase = torch.cumsum(w0 * torch.ones(128), dim=-1)
    x = torch.sin(phase).unsqueeze(0).float()

    # Test without windowing -- should be exact
    sc = SpectralCentroid(sr, scaling="none", window="none")
    y = sc(x)
    assert y.shape == (1,)
    torch.testing.assert_close(y, torch.tensor([f0]), atol=1e-4, rtol=1e-4)


def test_spectral_flatness():
    sr = 44100
    w0 = 2.0 * torch.pi * (440.0 / sr)
    phase = torch.cumsum(w0 * torch.ones(128), dim=-1)
    x = torch.sin(phase).unsqueeze(0).float()

    sf = SpectralFlatness(window="hann")
    y = sf(x)
    assert torch.all(y < -150.0)

    g = torch.Generator()
    g.manual_seed(0)
    noise = torch.rand(1, sr, generator=g) * 2.0 - 1.0
    y = sf(noise)
    assert torch.all(y > -6.0)

    silence = torch.zeros(1, 1024)
    y = sf(silence)
    torch.testing.assert_close(y, torch.zeros_like(y))


def test_feature_extractor():
    x = torch.zeros(100).unsqueeze(0)
    x[:, 50:] = 1.0
    features = [OnsetSegment(50), RMS()]
    extractor = FeatureExtractor(features)
    y = extractor(x)
    torch.testing.assert_close(y, torch.tensor([np.sqrt(1e-8)], dtype=torch.float))
