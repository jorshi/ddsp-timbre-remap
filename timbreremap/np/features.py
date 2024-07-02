"""
Audio feature extraction functions for numpy arrays.
"""
import numpy as np
import torch


class Loudness:
    def __init__(self, db: bool = False, eps: float = 1e-8):
        super().__init__()
        self.db = db
        self.eps = eps

    def __call__(self, frames: np.array):
        assert frames.ndim == 2

        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(frames), axis=1))

        # Convert to dB
        if self.db:
            rms = 20 * np.log10(rms + self.eps)

        return rms


class SpectralCentroid:
    def __init__(self):
        pass

    def __call__(self, frames: np.array):
        assert frames.ndim == 2

        # Calculate FFT
        X = np.fft.rfft(frames, axis=1)
        X = np.abs(X)

        # Normalize -- using the torch version for compatibility
        X_norm = torch.nn.functional.normalize(torch.from_numpy(X), p=1, dim=-1).numpy()

        # Calculate spectral centroid
        bins = np.arange(X.shape[1])
        spectral_centroid = np.sum(bins * X_norm, axis=1)

        return spectral_centroid
