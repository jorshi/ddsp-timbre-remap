import pytest
import torch
import torchaudio

from timbreremap.data import OnsetFeatureDataModule


def test_onset_feature_datamodule_init(tmp_path):
    audio_path = tmp_path / "test.wav"
    feature = torch.nn.Linear(44100, 3)
    onset_feature = torch.nn.Linear(44100, 1)
    data_module = OnsetFeatureDataModule(audio_path, feature, onset_feature)
    assert data_module.audio_path == audio_path
    assert data_module.feature == feature


@pytest.fixture
def audio_folder(tmp_path):
    """Create a folder with 8 noisy audio files"""
    for i in range(8):
        audio_path = tmp_path / f"{i}.wav"
        test_audio = torch.rand(1, 44100) * 2.0 - 1.0
        amp_env = torch.linspace(1, 0, 44100)
        test_audio = test_audio * amp_env
        torchaudio.save(audio_path, test_audio, 44100)

    yield tmp_path

    # Clean up
    for p in tmp_path.glob("*.wav"):
        p.unlink()


def test_onset_feature_datamodule_prepare(audio_folder):
    feature = torch.nn.Linear(44100, 6)
    onset_feature = torch.nn.Linear(44100, 3)
    data_module = OnsetFeatureDataModule(
        audio_folder, feature, onset_feature, sample_rate=44100
    )
    data_module.prepare_data()

    assert hasattr(data_module, "full_features")
    assert hasattr(data_module, "onset_features")
    assert data_module.full_features.shape == (8, 6)
    assert data_module.onset_features.shape == (8, 3)


def test_onset_feature_datamodule_setup(audio_folder):
    feature = torch.nn.Linear(44100, 6)
    onset_feature = torch.nn.Linear(44100, 3)
    data_module = OnsetFeatureDataModule(
        audio_folder, feature, onset_feature, sample_rate=44100, return_norm=True
    )
    data_module.prepare_data()
    data_module.setup("fit")

    assert hasattr(data_module, "train_dataset")
    assert len(data_module.train_dataset) == 8
    o, f, w = data_module.train_dataset[0]
    assert o.shape == (3,)
    assert f.shape == (6,)
    assert w.shape == f.shape
