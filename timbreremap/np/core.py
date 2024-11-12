"""
Core funcionality for the numpy backend.
"""
import numpy as np
from numba import jit
from scipy import signal


class HighPassFilter:
    """
    Simple implementation of a high-pass filter
    """

    def __init__(
        self, sr: int, cutoff: float, q: float = 0.707, peak_gain: float = 0.0
    ):
        self.sr = sr
        self.cutoff = cutoff
        self.q = q
        self.peak_gain = peak_gain
        self._init_filter()

    def _init_filter(self):
        K = np.tan(np.pi * self.cutoff / self.sr)
        norm = 1 / (1 + K / self.q + K * K)
        self.a0 = 1 * norm
        self.a1 = -2 * self.a0
        self.a2 = self.a0
        self.b1 = 2 * (K * K - 1) * norm
        self.b2 = (1 - K / self.q + K * K) * norm

    def __call__(self, x: np.array):
        assert x.ndim == 2 and x.shape[0] == 1
        y = signal.lfilter(
            [self.a0, self.a1, self.a2], [1, self.b1, self.b2], x, axis=1, zi=None
        )
        return y


@jit(nopython=True)
def envelope_follower(x: np.array, up: float, down: float, initial: float = 0.0):
    y = np.zeros_like(x)
    y0 = initial
    for i in range(y.shape[-1]):
        if x[0, i] > y0:
            y0 = up * (x[0, i] - y0) + y0
        else:
            y0 = down * (x[0, i] - y0) + y0
        y[0, i] = y0
    return y


class EnvelopeFollower:
    def __init__(self, attack_samples: int, release_samples: int):
        self.up = 1.0 / attack_samples
        self.down = 1.0 / release_samples

    def __call__(self, x: np.array, initial: float = 0.0):
        assert x.ndim == 2 and x.shape[0] == 1
        return envelope_follower(x, self.up, self.down, initial=initial)


@jit(nopython=True)
def detect_onset(x: np.array, on_thresh: float, off_thresh: float, wait: int):
    debounce = -1
    onsets = []
    for i in range(1, x.shape[-1]):
        if x[0, i] >= on_thresh and x[0, i - 1] < on_thresh and debounce == -1:
            onsets.append(i)
            debounce = wait

        if debounce > 0:
            debounce -= 1

        if debounce == 0 and x[0, i] < off_thresh:
            debounce = -1

    return onsets


class OnsetDetection:
    def __init__(
        self,
        sr: int,
        on_thresh: float = 16.0,
        off_thresh: float = 4.6666,
        wait: int = 1323,
        min_db: float = -55.0,
        eps: float = 1e-8,
    ):
        self.env_fast = EnvelopeFollower(3.0, 383.0)
        self.env_slow = EnvelopeFollower(2205.0, 2205.0)
        self.high_pass = HighPassFilter(sr, 600.0)
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.min_db = min_db
        self.wait = wait
        self.eps = eps

    def _onset_signal(self, x: np.array):
        # Filter
        x = self.high_pass(x)

        # Rectify, convert to dB, and set minimum value
        x = np.abs(x)
        x = 20 * np.log10(x + self.eps)
        x[x < self.min_db] = self.min_db

        # Calculate envelope
        env_fast = self.env_fast(x, initial=self.min_db)
        env_slow = self.env_slow(x, initial=self.min_db)
        diff = env_fast - env_slow

        return diff

    def __call__(self, x: np.array):
        assert x.ndim == 2 and x.shape[0] == 1, "Monophone audio only."

        # Calculate envelope
        onset = self._onset_signal(x)
        onsets = detect_onset(onset, self.on_thresh, self.off_thresh, self.wait)

        return onsets


class OnsetFrames:
    def __init__(
        self,
        sr: int,
        frame_size: int,
        pad_overlap: bool = True,  # Prevent overlap between frames with padding
        overlap_buffer: int = 32,  # Number of samples to look ahead for overlap
        backtrack: int = 0,  # Number of samples to backtrack for extraction
        **kwargs
    ):
        self.sr = sr
        self.frame_size = frame_size
        self.pad_overlap = pad_overlap
        self.overlap_buffer = overlap_buffer
        self.backtrack = backtrack
        self.onset = OnsetDetection(sr, **kwargs)

    def __call__(self, x: np.array):
        assert x.ndim == 2 and x.shape[0] == 1, "Monophone audio only."

        # Compute onsets
        onsets = self.onset(x)

        # Extract frames
        frames = []
        for j, onset in enumerate(onsets):
            # Extract the frame -- avoid overlap if pad_overlap is True
            start = max(onset - self.backtrack, 0)
            if (
                self.pad_overlap
                and j < len(onsets) - 1
                and (onsets[j + 1] - start < self.frame_size)
            ):
                frame = x[0, start : onsets[j + 1] - self.overlap_buffer]

                # Apply a fade out to the end of the frame
                fade = np.hanning(self.overlap_buffer * 2)[self.overlap_buffer :]
                frame[-self.overlap_buffer :] *= fade[
                    : frame.shape[-1] - self.overlap_buffer
                ]
            else:
                frame = x[0, start : start + self.frame_size]

            # Pad with zeros if necessary
            if frame.shape[-1] < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - frame.shape[-1]))

            assert frame.shape[-1] == self.frame_size
            frames.append(frame)

        return np.array(frames)


class KWeightingFilter:
    """
    K-Weighting Filter
    """

    def __init__(self, sr: int):
        self.sr = sr
        self._init_filter()

    def _init_filter(self):
        # Shelving filter coefficients
        f0 = 1681.974450955533
        G = 3.999843853973347
        Q = 0.7071752369554196

        K = np.tan(np.pi * f0 / self.sr)
        Vh = np.power(10.0, G / 20.0)
        Vb = np.power(Vh, 0.4996667741545416)

        shelvB = np.zeros(3)
        shelvA = np.zeros(3)
        shelvA[0] = 1.0

        a0 = 1.0 + K / Q + K * K
        shelvB[0] = (Vh + Vb * K / Q + K * K) / a0
        shelvB[1] = 2.0 * (K * K - Vh) / a0
        shelvB[2] = (Vh - Vb * K / Q + K * K) / a0
        shelvA[1] = 2.0 * (K * K - 1.0) / a0
        shelvA[2] = (1.0 - K / Q + K * K) / a0

        self.shelvB = shelvB
        self.shelvA = shelvA

        # High-pass filter coefficients
        f0 = 38.13547087602444
        Q = 0.5003270373238773
        K = np.tan(np.pi * f0 / self.sr)

        hiB = np.array([0.9946, -1.9892,  0.9946])
        hiA = np.array([1.0, 0.0, 0.0])

        hiA[1] = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
        hiA[2] = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)

        self.hiB = hiB
        self.hiA = hiA
    
    def __call__(self, x: np.array):
        assert x.ndim == 2 and x.shape[0] == 1
        y = signal.lfilter(self.shelvB, self.shelvA, x, axis=1, zi=None)
        y = signal.lfilter(self.hiB, self.hiA, y, axis=1, zi=None)
        return y
