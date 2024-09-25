"""
Differentiable Audio Features
"""
from collections import OrderedDict
from typing import List
from typing import Literal

import numpy as np
import pyloudnorm
import torch
import torchaudio


class FeatureExtractor(torch.nn.Module):
    """
    A serial connection of feature extraction layers.
    """

    def __init__(self, features: List[torch.nn.Module]):
        super().__init__()
        self.features = torch.nn.Sequential(*features)

    def forward(self, x: torch.Tensor):
        return self.features(x)


class FeatureCollection(torch.nn.Module):
    """
    A collection of feature extractors that are flattened
    """

    def __init__(self, features: List[torch.nn.Module]):
        super().__init__()
        self.features = torch.nn.ModuleList(features)

    def forward(self, x: torch.Tensor):
        results = []
        for feature in self.features:
            results.append(feature(x))
        return torch.cat(results, dim=-1)


class NumpyWrapper(torch.nn.Module):
    """
    Wrap a feature extractor for numpy inputs.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x: np.ndarray):
        return self.func(torch.from_numpy(x).float()).numpy()


class OnsetSegment(torch.nn.Module):
    """
    Segment a signal in relation to the start of the signal.
    """

    def __init__(self, window: int = 2048, delay: int = 0):
        super().__init__()
        self.window = window
        self.delay = delay

    def forward(self, x: torch.Tensor):
        return x[..., self.delay : self.delay + self.window]


class CascadingFrameExtactor(torch.nn.Module):
    """
    Given frames. Computes features and computes summary statistics
    over an increasing number of frames from the onset
    """

    def __init__(
        self,
        extractors: List[torch.nn.Module],
        num_frames: list[int],
        frame_size: int = 2048,
        hop_size: int = 1024,
        pad_start: int = None,
        include_mean: bool = True,
        include_diff: bool = False,
        always_from_onset: bool = False,
    ):
        super().__init__()
        self.extractors = extractors
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.pad_start = pad_start
        self.include_mean = include_mean
        self.include_diff = include_diff
        self.always_from_onset = always_from_onset
        self.flattened_features = []

        frame_feature_names = []
        k = 0
        for n in num_frames:
            if include_mean:
                frame_feature_names.append(f"{k}_{n}_mean")
            if include_diff and n > 1:
                frame_feature_names.append(f"{k}_{n}_diff")
            k = k if always_from_onset else k + n

        for extractor in self.extractors:
            ename = extractor._get_name()
            for n in frame_feature_names:
                self.flattened_features.append((ename, n))

        self.num_features = len(self.flattened_features)

    def forward(
        self,
        x: torch.Tensor,  # (batch, samples)
    ):
        y = self.get_as_dict(x)
        flattened_y = []
        for extractor, feature in self.flattened_features:
            flattened_y.append(y[extractor][feature].unsqueeze(-1))

        return torch.cat(flattened_y, dim=-1)

    def get_as_dict(self, x: torch.Tensor):
        """
        Returns a dictionary of features
        """
        if self.pad_start is not None:
            x = torch.nn.functional.pad(x, (self.pad_start, 0))

        x = x.unfold(-1, self.frame_size, self.hop_size)
        assert x.ndim == 3
        assert x.shape[1] >= sum(self.num_frames)

        results = OrderedDict()
        for extractor in self.extractors:
            ename = extractor._get_name()
            if ename not in results:
                results[ename] = OrderedDict()

            k = 0
            for n in self.num_frames:
                frames = x[..., k : k + n, :]
                y = extractor(frames)

                if self.include_mean:
                    y_mean = y.mean(dim=-1)
                    results[ename][f"{k}_{n}_mean"] = y_mean

                if self.include_diff and n > 1:
                    y_diff = torch.diff(y, dim=-1)
                    y_diff_mean = y_diff.mean(dim=-1)
                    results[ename][f"{k}_{n}_diff"] = y_diff_mean

                k = k if self.always_from_onset else k + n

        return results


class RMS(torch.nn.Module):
    """
    Root mean square of a signal.
    """

    def __init__(
        self,
        db: bool = False,  # Convert to dB
    ):
        super().__init__()
        self.db = db

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1) + 1e-8)
        if self.db:
            rms = 20 * torch.log10(rms + 1e-8)
            # Clipping isn't good for gradients ?
            # rms = torch.clamp(rms, min=-120.0)
        return rms


class Loudness(torch.nn.Module):
    """
    Computes loudness (LKFS) by applying K-weighting filters based on ITU-R BS.1770-4
    """

    def __init__(
        self,
        sample_rate: int,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.epsilon = epsilon

        # Setup K-weighting filters
        a_coefs = []
        b_coefs = []
        filters = pyloudnorm.Meter(sample_rate, "K-weighting")._filters
        for filt in filters.items():
            a_coefs.append(filt[1].a)
            b_coefs.append(filt[1].b)

        a_coefs = np.array(a_coefs)
        b_coefs = np.array(b_coefs)
        self.register_buffer("a_coefs", torch.tensor(a_coefs, dtype=torch.float))
        self.register_buffer("b_coefs", torch.tensor(b_coefs, dtype=torch.float))

    def prefilter(self, x: torch.Tensor):
        """
        Prefilter the signal with K-weighting filters.
        """
        # Apply K-weighting filters in series
        a_coefs = self.a_coefs.to(x.device).split(1, dim=0)
        b_coefs = self.b_coefs.to(x.device).split(1, dim=0)
        for a, b in zip(a_coefs, b_coefs):
            x = torchaudio.functional.lfilter(x, a.squeeze(), b.squeeze())

        return x

    def forward(self, x: torch.Tensor):
        """
        Compute loudness (LKFS) of a signal.
        """
        x = self.prefilter(x)
        loudness = torch.mean(torch.square(x), dim=-1)
        loudness = -0.691 + 10.0 * torch.log10(loudness + self.epsilon)
        return loudness


class SpectralCentroid(torch.nn.Module):
    """
    Spectral centroid of a signal.
    """

    def __init__(
        self,
        sample_rate: int,
        window: Literal["hann", "flat_top", "none"] = "hann",
        compress: bool = False,
        floor: float = None,  # Floor spectral magnitudes to this value
        scaling: Literal["semitone", "kazazis", "none"] = "semitone",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.scaling = scaling
        self.window_fn = get_window_fn(window)
        self.compress = compress
        self.floor = floor

    def forward(self, x: torch.Tensor):
        # Apply a window
        if self.window_fn is not None:
            window = self.window_fn(x.shape[-1], device=x.device)
            x = x * window

        # Calculate FFT
        X = torch.fft.rfft(x, dim=-1)
        X = torch.abs(X)

        if self.floor is not None:
            X = torch.where(X < self.floor, self.floor, X)

        # Compression
        if self.compress:
            X = torch.log(1 + X)

        X_norm = torch.nn.functional.normalize(X, p=1, dim=-1)

        # Calculate spectral centroid
        bins = torch.arange(X.shape[-1], device=x.device)
        spectral_centroid = torch.sum(bins * X_norm, dim=-1)

        # Convert to Hz
        bin_hz = self.sample_rate / x.shape[-1]
        spectral_centroid = spectral_centroid * bin_hz

        # Convert to semitones
        if self.scaling == "semitone":
            spectral_centroid = (
                12 * torch.log2((spectral_centroid + 1e-8) / 440.0) + 69.0
            )
        elif self.scaling == "kazazis":
            spectral_centroid = -34.61 * torch.pow(spectral_centroid, -0.1621) + 21.2985
            spectral_centroid = torch.nan_to_num(spectral_centroid, nan=0.0, posinf=0.0, neginf=0.0)

        return spectral_centroid


class SpectralSpread(torch.nn.Module):
    """
    Spectral spread of a signal.

    TODO: there is a lot of repeated code here with SpectralCentroid, and in general
    with the spectral features. This should be refactored to avoid repeated calls to
    FFTs etc.
    """

    def __init__(
        self,
        window: Literal["hann", "flat_top", "none"] = "hann",
        compress: bool = False,
        floor: float = None,  # Floor spectral magnitudes to this value
    ):
        super().__init__()
        self.window_fn = get_window_fn(window)
        self.compress = compress
        self.floor = floor

    def forward(self, x: torch.Tensor):
        # Apply a window
        if self.window_fn is not None:
            window = self.window_fn(x.shape[-1], device=x.device)
            x = x * window

        # Calculate FFT
        X = torch.fft.rfft(x, dim=-1)
        X = torch.abs(X)

        if self.floor is not None:
            X = torch.where(X < self.floor, self.floor, X)

        # Compression
        if self.compress:
            X = torch.log(1 + X)

        X_norm = torch.nn.functional.normalize(X, p=1, dim=-1)

        # Calculate spectral centroid
        bins = torch.arange(X.shape[-1], device=x.device)
        spectral_centroid = torch.sum(bins * X_norm, dim=-1)

        # Calculate spectral spread
        spectral_spread = torch.sum(
            torch.square(bins - spectral_centroid[..., None]) * X_norm, dim=-1
        )

        return spectral_spread


class SpectralFlatness(torch.nn.Module):
    """
    Spectral flatness of a signal.
    """

    def __init__(
        self,
        amin: float = 1e-10,
        window: Literal["hann", "flat_top", "none"] = "hann",
        compress: bool = False,
    ) -> None:
        super().__init__()
        self.amin = amin
        self.window_fn = get_window_fn(window)
        self.compress = compress

    def forward(self, x: torch.Tensor):
        # Apply a window
        if self.window_fn is not None:
            window = self.window_fn(x.shape[-1], device=x.device)
            x = x * window

        # Calculate FFT
        X = torch.fft.rfft(x, dim=-1)
        X = torch.abs(X)

        # Compression
        if self.compress:
            X = torch.log(1 + X)

        X_power = torch.where(X**2.0 < self.amin, self.amin, X**2.0)
        gmean = torch.exp(torch.mean(torch.log(X_power), dim=-1))
        amean = torch.mean(X_power, dim=-1)

        # Calculate spectral flatness
        spectral_flatness = gmean / amean

        # Convert to dB
        spectral_flatness = 20.0 * torch.log10(spectral_flatness + 1e-8)

        return spectral_flatness


class SpectralFlux(torch.nn.Module):
    """
    Spectral flux of a signal.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3
        assert x.shape[1] > 1, "Must have at least two frames"

        # Apply a window
        window = torch.hann_window(x.shape[-1], device=x.device)
        x = x * window

        # Calculate FFT
        X = torch.fft.rfft(x, dim=-1)
        X = torch.abs(X)

        flux = torch.diff(X, dim=-2)
        flux = (flux + torch.abs(flux)) / 2
        flux = torch.square(flux)
        flux = torch.sum(flux, dim=-1)

        return flux


class AmplitudeEnvelope(torch.nn.Module):
    """
    Get the amplitude envelope of a signal by convolution with a window.
    """

    def __init__(self, window: int = 2048):
        super().__init__()
        self.window = window

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3

        # Calculate the amplitude envelope
        window = torch.hann_window(self.window, device=x.device)
        window = window[None, None, :]

        x = torch.square(x)
        y = torch.nn.functional.conv1d(x, window, padding="same")
        assert torch.all(torch.isfinite(y))

        return y


class TemporalCentroid(torch.nn.Module):
    """
    Temporal centroid of a signal.
    """

    def __init__(
        self,
        sample_rate: int,
        window_size: int = 2048,
        scaling: Literal["schlauch", "none"] = "none",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.envelope = AmplitudeEnvelope(window=window_size)
        self.scaling = scaling

    def forward(self, x: torch.Tensor):
        env = self.envelope(x)
        y = torch.sum(env * torch.arange(env.shape[-1], device=x.device), dim=-1)
        y = y / (torch.sum(env, dim=-1) + 1e-8)
        y = y / self.sample_rate * 1000.0
        if self.scaling == "schlauch":
            y = 0.03 * torch.pow(y, 1.864)
        return y


class MFCC(torch.nn.Module):
    """
    Mel-frequency cepstral coefficients.
    """

    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int = 13,
        n_mels: int = 128,
        window: Literal["hann", "flat_top", "none"] = "hann",
        compress: bool = False,
        floor: float = None,  # Floor spectral magnitudes to this value
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.window_fn = get_window_fn(window)
        self.compress = compress
        self.floor = floor

    def forward(self, x: torch.Tensor):
        # Apply a window
        if self.window_fn is not None:
            window = self.window_fn(x.shape[-1], device=x.device)
            x = x * window

        # Calculate FFT
        X = torch.fft.rfft(x, dim=-1)
        X = torch.abs(X)

        fb = torchaudio.functional.melscale_fbanks(
            X.shape[-1], 0.0, self.sample_rate / 2, self.n_mels, self.sample_rate
        )
        mel_scale = torch.matmul(X, fb.to(X.device))
        mel_scale = torch.log(mel_scale + 1e-6)

        dct_mat = torchaudio.functional.create_dct(
            self.n_mfcc, self.n_mels, norm="ortho"
        )
        mfcc = torch.matmul(mel_scale, dct_mat.to(mel_scale.device))

        return mfcc[..., 1:]


class NormSum(torch.nn.Module):
    """
    Sum of the normalized signal.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2
        assert not torch.all(torch.isnan(x))

        norm = x / torch.max(torch.abs(x))
        norm = torch.nan_to_num(norm, nan=0.0)
        y = torch.sum(norm, dim=-1)
        y = y / float(x.shape[-1])
        return y


class AmpEnvSum(FeatureExtractor):
    """
    Sum of the amplitude envelope.
    """

    def __init__(self, window: int = 2048):
        super().__init__([AmplitudeEnvelope(window), NormSum()])


def get_window_fn(window: str):
    if window == "hann":
        return torch.hann_window
    elif window == "flat_top":
        return flat_top_window
    elif window == "none":
        return None
    else:
        raise ValueError(f"Unknown window type: {window}")


def flat_top_window(size, device="cpu"):
    """
    Flat top window for spectral analysis.
    https://en.wikipedia.org/wiki/Window_function#Flat_top_window
    """
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368

    n = torch.arange(size, dtype=torch.float, device=device)
    window = (
        a0
        - a1 * torch.cos(2 * torch.pi * n / (size - 1))
        + a2 * torch.cos(4 * torch.pi * n / (size - 1))
        - a3 * torch.cos(6 * torch.pi * n / (size - 1))
        + a4 * torch.cos(8 * torch.pi * n / (size - 1))
    )
    return window
