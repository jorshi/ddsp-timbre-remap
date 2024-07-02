import numpy as np

from timbreremap.np.core import envelope_follower
from timbreremap.np.core import EnvelopeFollower
from timbreremap.np.core import HighPassFilter


def test_construct_hpf():
    filter = HighPassFilter(sr=44100, cutoff=1000)
    assert filter.sr == 44100
    assert filter.cutoff == 1000


def test_hpf_call():
    x = np.random.randn(1, 44100)
    filter = HighPassFilter(sr=44100, cutoff=10000)
    y = filter(x)
    assert y.shape == x.shape


def test_envelope_follower():
    # Test envelope follower rising up
    x = np.zeros((1, 1000))
    x[0, 500:] = 1.0
    y = envelope_follower(x, up=1.0 / 100.0, down=1.0)
    assert y.shape == x.shape
    assert np.all(y[:, :500] == 0.0)
    assert np.all(y[:, 500:600] > 0.0) and np.all(y[:, 500:599] < (1.0 - 1.0 / np.e))
    assert np.all(y[:, 600:] >= (1.0 - 1.0 / np.e))

    # Test envelope follower falling down
    x = np.zeros((1, 1000))
    x[0, :500] = 1.0
    y = envelope_follower(x, up=1.0, down=1.0 / 100.0)
    assert y.shape == x.shape
    assert np.all(y[:, :500] == 1.0)
    assert np.all(y[:, 500:600] < 1.0) and np.all(y[:, 500:599] > 1.0 / np.e)
    assert np.all(y[:, 600:] <= 1.0 / np.e)


def test_envelope_follower_class(mocker):
    follower = EnvelopeFollower(attack_samples=100, release_samples=100)
    assert follower.up == 0.01
    assert follower.down == 0.01

    x = np.zeros((1, 1000))
    x[0, 500:600] = 1.0

    mocked_ef = mocker.patch("timbreremap.np.core.envelope_follower")
    follower(x)
    mocked_ef.assert_called_once_with(x, 0.01, 0.01, initial=0.0)
