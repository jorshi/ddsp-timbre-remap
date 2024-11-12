from .feature import CascadingFrameExtactor
from .feature import FeatureCollection
from .feature import Loudness
from .feature import SpectralCentroid
from .feature import SpectralFlatness


def simple_spectral_factory(sample_rate: int):
    return FeatureCollection(
        [SpectralCentroid(sample_rate=sample_rate), SpectralFlatness()]
    )


def spectral_loudness_factory(sample_rate: int, num_samples: int):
    centroid = SpectralCentroid(sample_rate=sample_rate)
    loudness = Loudness(sample_rate=sample_rate)

    frame_size = 2048
    hop_size = 512
    num_frames = num_samples // hop_size
    print(num_frames)

    extractor = CascadingFrameExtactor(
        [centroid, loudness],
        num_frames=[
            num_frames,
        ],
        frame_size=frame_size,
        hop_size=hop_size,
        include_diff=True,
    )
    return extractor
