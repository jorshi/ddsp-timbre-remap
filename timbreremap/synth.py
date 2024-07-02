"""
Modules for synthesizing drum sounds

TODO: should we use torchsynth for this?
"""
import json
from collections import OrderedDict
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torchaudio
from einops import repeat


class ParamaterNormalizer:
    """
    Holds min and max values for a parameter and provides methods for normalizing
    between 0 and 1 and vice versa
    """

    def __init__(self, min_value, max_value, description=None):
        self.min_value = min_value
        self.max_value = max_value
        self.description = description

    def __repr__(self):
        return (
            f"ParamaterNormalizer(Min: {self.min_value}, Max: {self.max_value}, "
            f"Desc: {self.description})"
        )

    def from_0to1(self, x):
        return self.min_value + (self.max_value - self.min_value) * x

    def to_0to1(self, x):
        return (x - self.min_value) / (self.max_value - self.min_value)


class AbstractModule(torch.nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.normalizers = OrderedDict()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ExpDecayEnvelope(AbstractModule):
    """
    Exponential decay envelope
    C++ version: ExpDecayEnvelope
    """

    def __init__(
        self, sample_rate: int, decay_min: float = 10.0, decay_max: float = 2000.0
    ):
        super().__init__(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.normalizers["decay"] = ParamaterNormalizer(
            decay_min, decay_max, "decay time ms"
        )
        self.attack_samples = int(0.001 * self.sample_rate)
        self.attack_incr = 1.0 / self.attack_samples

    def forward(self, num_samples: int, decay: torch.Tensor):
        assert decay.ndim == 2
        assert decay.shape[1] == 1
        assert (
            decay.min() >= 0.0 and decay.max() <= 1.0
        ), "param must be between 0 and 1"

        # Calculated the samplewise decay rate
        decay_ms = self.normalizers["decay"].from_0to1(decay)
        decay_samples = decay_ms * self.sample_rate / 1000.0
        decay_rate = 1.0 - (6.91 / decay_samples)

        # Calculate the decay envelope
        decay_samples = num_samples - self.attack_samples
        assert decay_samples > 0, "num_samples must be greater than attack_samples"

        env = torch.ones(decay_rate.shape[0], decay_samples, device=decay_rate.device)
        env[:, 1:] = decay_rate
        env = torch.cumprod(env, dim=-1)

        # Add attack
        attack = torch.ones(
            decay_rate.shape[0], self.attack_samples, device=decay_rate.device
        )
        attack = torch.cumsum(attack * self.attack_incr, dim=-1)

        # Combine attack and decay
        env = torch.cat((attack, env), dim=-1)
        return env


class ExponentialDecay(AbstractModule):
    """
    Exponential decay envelope
    """

    def __init__(self, sample_rate: int):
        super().__init__(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.normalizers["decay"] = ParamaterNormalizer(10.0, 2000.0, "decay time ms")

    def forward(self, num_samples: int, decay: torch.Tensor):
        assert decay.ndim == 2
        assert decay.shape[1] == 1

        # Calculated the samplewise decay rate
        decay_ms = self.normalizers["decay"].from_0to1(decay)
        decay_samples = decay_ms * self.sample_rate / 1000.0
        decay_rate = 1.0 - (6.91 / decay_samples)

        # Calculate the envelope
        env = torch.ones(decay_rate.shape[0], num_samples, device=decay_rate.device)
        env[:, 1:] = decay_rate
        env = torch.cumprod(env, dim=-1)

        return env


class SinusoidalOscillator(AbstractModule):
    """
    A sinusoidal oscillator
    C++ version: SinusoidalOscillator

    TODO: slight numerical differences between cpp and py versions when using modulation
    """

    def __init__(self, sample_rate: int):
        super().__init__(sample_rate=sample_rate)
        self.normalizers["freq"] = ParamaterNormalizer(20.0, 2000.0, "frequency (Hz)")
        self.normalizers["mod"] = ParamaterNormalizer(
            -1.0, 2.0, "freq envelope amount (ratio)"
        )

    def forward(
        self,
        num_samples: int,
        freq: torch.Tensor,
        mod_env: torch.Tensor,
        mod_amount: torch.Tensor,
    ):
        assert freq.ndim == 2
        assert freq.shape[1] == 1
        assert mod_amount.shape == freq.shape
        assert mod_env.ndim == 2
        assert mod_env.shape[1] == num_samples

        # Calculate the phase
        f0 = self.normalizers["freq"].from_0to1(freq)
        f0 = 2 * torch.pi * f0 / self.sample_rate

        freq_env = torch.ones(f0.shape[0], num_samples, device=f0.device) * f0

        mod_amount = self.normalizers["mod"].from_0to1(mod_amount)
        mod_env = mod_env * mod_amount * f0
        freq_env = freq_env + mod_env

        # Add a zero to the beginning of the envelope for zero intial phase
        freq_env = torch.cat((torch.zeros_like(freq_env[:, :1]), freq_env), dim=-1)

        # Integrate to get the phhase
        phase = torch.cumsum(freq_env, dim=-1)[:, :-1]

        # Generate the signal
        y = torch.sin(phase)

        return y


class Tanh(AbstractModule):
    """
    tanh waveshaper
    C++ version: Tanh
    """

    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)
        self.normalizers["in_gain"] = ParamaterNormalizer(
            -24.0, 24.0, "input gain (db)"
        )

    def forward(self, x: torch.Tensor, in_gain: torch.Tensor):
        in_gain = self.normalizers["in_gain"].from_0to1(in_gain)
        in_gain = torch.pow(10.0, in_gain / 20.0)
        return torch.tanh(in_gain * x)


class Gain(AbstractModule):
    """
    Gain module with a gain parameter in decibels
    C++ version: Gain
    """

    def __init__(
        self, sample_rate: int, min_gain: float = -60.0, max_gain: float = 6.0
    ):
        super().__init__(sample_rate)
        self.normalizers["gain"] = ParamaterNormalizer(min_gain, max_gain, "gain (db)")

    def forward(self, x: torch.Tensor, gain: torch.Tensor):
        gain = self.normalizers["gain"].from_0to1(gain)
        gain = torch.pow(10.0, gain / 20.0)
        return gain * x


class WhiteNoise(AbstractModule):
    """
    White noise generator
    C++ version: WhiteNoise
    """

    def __init__(
        self,
        sample_rate: int,
        buffer_noise: bool = False,
        buffer_size: int = 0,
        device: torch.device = "cpu",
    ):
        super().__init__(sample_rate)
        if buffer_noise:
            assert buffer_size > 0, "buffer_size must be greater than 0"
            noise = torch.rand(1, buffer_size, device=device) * 2.0 - 1.0
            self.register_buffer("noise", noise)

    def forward(self, batch_size: int, num_samples: int, device: torch.device):
        if hasattr(self, "noise"):
            y = repeat(self.noise, "1 n -> b n", b=batch_size)
            y = y[:, :num_samples]
        else:
            y = torch.rand(batch_size, num_samples, device=device) * 2.0 - 1.0
        return y


class CrossFade(AbstractModule):
    """
    Cross fade between two signals
    C++ version: CrossFade
    """

    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)
        self.normalizers["fade"] = ParamaterNormalizer(0.0, 1.0, "fade amount")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, fade: torch.Tensor):
        fade = self.normalizers["fade"].from_0to1(fade)
        return torch.sqrt(fade) * x1 + torch.sqrt((1.0 - fade)) * x2


class Biquad(AbstractModule):
    """
    Biquad Filter with Butterworth coefficients
    """

    def __init__(self, sample_rate: int, filter_type: str = "lowpass"):
        super().__init__(sample_rate)

        # Only lowpass and highpass are supported for now
        self.filter_type = filter_type
        assert self.filter_type in ["lowpass", "highpass"]

        self.normalizers["freq"] = ParamaterNormalizer(
            20.0, sample_rate // 2, "cutoff freq"
        )
        self.normalizers["q"] = ParamaterNormalizer(0.5, 10.0, "q")

    def lowpass_coefficients(self, freq: torch.Tensor, q: torch.Tensor):
        """
        Calculate the coefficients for a lowpass filter
        """
        freq = torch.tan(torch.pi * freq / self.sample_rate)
        norm = 1 / (1 + freq / q + freq * freq)
        a0 = freq * freq * norm
        a1 = 2 * a0
        a2 = a0
        b1 = 2 * (freq * freq - 1) * norm
        b2 = (1 - freq / q + freq * freq) * norm
        a_coefs = torch.cat((a0, a1, a2), dim=-1)
        b_coefs = torch.cat((torch.ones_like(a0), b1, b2), dim=-1)
        return a_coefs, b_coefs

    def highpass_coefficients(self, freq: torch.Tensor, q: torch.Tensor):
        """
        Calculate the coefficients for a highpass filter
        """
        freq = torch.tan(torch.pi * freq / self.sample_rate)
        norm = 1 / (1 + freq / q + freq * freq)
        a0 = 1 * norm
        a1 = -2 * a0
        a2 = a0
        b1 = 2 * (freq * freq - 1) * norm
        b2 = (1 - freq / q + freq * freq) * norm
        a_coefs = torch.cat((a0, a1, a2), dim=-1)
        b_coefs = torch.cat((torch.ones_like(a0), b1, b2), dim=-1)
        return a_coefs, b_coefs

    def forward(self, x: torch.Tensor, freq: torch.Tensor, q: torch.Tensor):
        freq = self.normalizers["freq"].from_0to1(freq)
        q = self.normalizers["q"].from_0to1(q)

        if self.filter_type == "lowpass":
            a_coefs, b_coefs = self.lowpass_coefficients(freq, q)
        elif self.filter_type == "highpass":
            a_coefs, b_coefs = self.highpass_coefficients(freq, q)

        # Order of coefficients is reversed from documentation
        y = torchaudio.functional.lfilter(x, b_coefs, a_coefs, batching=True)
        return y


class AbstractSynth(torch.nn.Module):
    """
    Abstract synthesizer class
    """

    def __init__(self, sample_rate: int, num_samples: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def forward(self, params: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def get_param_dict(self):
        """
        Returns a dictionary of parameters and their normalizers
        """
        param_dict = OrderedDict()
        for name, module in self.named_modules():
            if not hasattr(module, "normalizers"):
                continue
            for param_name, normalizer in module.normalizers.items():
                param_dict[(name, param_name)] = normalizer
        return param_dict

    def get_num_params(self):
        """
        Returns the number of parameters in the synthesizer
        """
        return len(self.get_param_dict())

    def params_from_dict(self, param_dict: Dict[str, Union[float, torch.Tensor]]):
        """
        Converts a dictionary of parameter values to a tensor of parameters that
        are normalized between 0 and 1
        """
        normalizers = self.get_param_dict()
        params = []
        for key, value in param_dict.items():
            normalizer = normalizers[key]
            if isinstance(value, float):
                value = torch.tensor([value])
            params.append(normalizer.to_0to1(value))

        return torch.vstack(params).T

    def damping_from_dict(self, damping_dict: Dict[str, Union[float, torch.Tensor]]):
        damping = []
        for name, module in self.named_modules():
            if not hasattr(module, "normalizers"):
                continue
            for param_name, _ in module.normalizers.items():
                key = (name, param_name)
                if key in damping_dict:
                    damping.append(damping_dict[key])
                else:
                    damping.append(1.0)

        return torch.tensor(damping).unsqueeze(0)

    @staticmethod
    def save_params_json(
        path: str,
        patch: Dict[Tuple[str, str], float],
        damping: Dict[Tuple[str, str], float] = None,
    ):
        param_json = {"preset": {}}
        for k, v in patch.items():
            assert "." not in k[0], "Parameter names cannot contain '.'"
            assert "." not in k[1], "Parameter names cannot contain '.'"
            new_key = k[0] + "." + k[1]
            param_json["preset"][new_key] = v

        if damping is not None:
            param_json["damping"] = {}
            for k, v in damping.items():
                assert "." not in k[0], "Parameter names cannot contain '.'"
                assert "." not in k[1], "Parameter names cannot contain '.'"
                new_key = k[0] + "." + k[1]
                param_json["damping"][new_key] = v

        with open(path, "w") as f:
            json.dump(param_json, f, indent=4)

    def load_params_json(self, path: str, as_tensor: bool = True):
        with open(path, "r") as f:
            param_json = json.load(f)

        patch = {}
        for k, v in param_json["preset"].items():
            module_name, param_name = k.split(".")
            patch[(module_name, param_name)] = v

        damping = {}
        if "damping" in param_json:
            for k, v in param_json["damping"].items():
                module_name, param_name = k.split(".")
                damping[(module_name, param_name)] = v

        # Normalize into tensors
        if as_tensor:
            patch = self.params_from_dict(patch)
            damping = self.damping_from_dict(damping)

        return patch, damping


class SimpleDrumSynth(AbstractSynth):
    def __init__(
        self, sample_rate, num_samples: int, buffer_noise=False, buffer_size=0
    ):
        super().__init__(sample_rate=sample_rate, num_samples=num_samples)
        self.amp_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.freq_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.osc = SinusoidalOscillator(sample_rate=sample_rate)
        self.tanh = Tanh(sample_rate=sample_rate)
        self.gain = Gain(sample_rate=sample_rate)
        self.noise_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.noise = WhiteNoise(
            sample_rate=sample_rate, buffer_noise=buffer_noise, buffer_size=buffer_size
        )
        self.tonal_gain = Gain(sample_rate=sample_rate)
        self.noise_gain = Gain(sample_rate=sample_rate)

    def forward(self, params: torch.Tensor, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = self.num_samples

        # Split params -- These should be the same order as the normalizers,
        # which is defined by the order returned by get_param_dict()
        # TODO: this is easy to mess up, is there a better way?
        assert params.shape[-1] == self.get_num_params()
        (
            decay,
            freq_decay,
            freq,
            freq_mod,
            in_gain,
            out_gain,
            noise_decay,
            tonal_gain,
            noise_gain,
        ) = torch.split(params, 1, dim=-1)

        freq_env = self.freq_env(num_samples, freq_decay)

        # Generate signal
        y = self.osc(num_samples, freq, freq_env, freq_mod)

        # Generate envelope
        env = self.amp_env(num_samples, decay)

        # Apply envelope
        y = y * env

        # Generate noise
        noise = self.noise(params.shape[0], num_samples, device=params.device)
        noise_env = self.noise_env(num_samples, noise_decay)
        noise = noise * noise_env
        noise = self.noise_gain(noise, noise_gain)

        # Add noise
        y = self.tonal_gain(y, tonal_gain)
        y = y + noise

        y = self.tanh(y, in_gain)
        y = self.gain(y, out_gain)

        return y


class FMDrumSynth(AbstractSynth):
    """
    FM drum synthesizer
    """

    def __init__(
        self, sample_rate, num_samples: int, buffer_noise=False, buffer_size=0
    ):
        super().__init__(sample_rate=sample_rate, num_samples=num_samples)
        self.amp_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.freq_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.osc = SinusoidalOscillator(sample_rate=sample_rate)
        self.mod_amp_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.mod_freq_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.mod_osc = SinusoidalOscillator(sample_rate=sample_rate)
        self.mod_gain = Gain(sample_rate, -48.0, 48.0)
        self.noise_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.noise = WhiteNoise(
            sample_rate=sample_rate, buffer_noise=buffer_noise, buffer_size=buffer_size
        )
        self.tonal_gain = Gain(sample_rate=sample_rate)
        self.noise_gain = Gain(sample_rate=sample_rate)
        self.tanh = Tanh(sample_rate=sample_rate)

    def forward(self, params: torch.Tensor, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = self.num_samples
        # Split params -- These should be the same order as the normalizers,
        # which is defined by the order returned by get_param_dict()
        assert params.shape[-1] == self.get_num_params()
        (
            amp_decay,
            freq_decay,
            freq,
            osc_mod,
            mod_amp_decay,
            mod_freq_decay,
            mod_freq,
            mod_osc_mod,
            mod_gain,
            noise_decay,
            tonal_gain,
            noise_gain,
            tanh_gain,
        ) = torch.split(params, 1, dim=-1)

        mod_freq_env = self.mod_freq_env(num_samples, mod_freq_decay)
        mod_amp_decay = self.mod_amp_env(num_samples, mod_amp_decay)
        y_mod = self.mod_osc(num_samples, mod_freq, mod_freq_env, mod_osc_mod)
        y_mod = y_mod * mod_amp_decay
        y_mod = self.mod_gain(y_mod, mod_gain)

        freq_env = self.freq_env(num_samples, freq_decay)
        amp_env = self.amp_env(num_samples, amp_decay)
        y = self.osc(num_samples, freq, freq_env + y_mod, osc_mod)
        y = y * amp_env

        # Generate noise
        noise = self.noise(params.shape[0], num_samples, device=params.device)
        noise_env = self.noise_env(num_samples, noise_decay)
        noise = noise * noise_env
        noise = self.noise_gain(noise, noise_gain)

        # Add noise
        y = self.tonal_gain(y, tonal_gain)
        y = y + noise

        y = self.tanh(y, tanh_gain)

        return y


class Snare808(AbstractSynth):
    def __init__(
        self, sample_rate, num_samples: int, buffer_noise=False, buffer_size=0
    ):
        super().__init__(sample_rate=sample_rate, num_samples=num_samples)
        self.osc1 = SinusoidalOscillator(sample_rate=sample_rate)
        self.osc2 = SinusoidalOscillator(sample_rate=sample_rate)
        self.freq_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.osc1_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.osc2_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.noise = WhiteNoise(
            sample_rate=sample_rate, buffer_noise=buffer_noise, buffer_size=buffer_size
        )
        self.noise_env = ExpDecayEnvelope(sample_rate=sample_rate)
        self.noise_filter = Biquad(sample_rate=sample_rate, filter_type="highpass")
        self.osc1_gain = Gain(sample_rate=sample_rate)
        self.osc2_gain = Gain(sample_rate=sample_rate)
        self.noise_gain = Gain(sample_rate=sample_rate)
        self.tanh = Tanh(sample_rate=sample_rate)

    def forward(self, params: torch.Tensor, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = self.num_samples

        # Split params -- These should be the same order as the normalizers,
        # which is defined by the order returned by get_param_dict()
        assert params.shape[-1] == self.get_num_params()
        (
            osc1_freq,
            osc1_mod,
            osc2_freq,
            osc2_mod,
            freq_decay,
            osc1_decay,
            osc2_decay,
            noise_decay,
            noise_freq,
            noise_q,
            osc1_gain,
            osc2_gain,
            noise_gain,
            tanh_gain,
        ) = torch.split(params, 1, dim=-1)

        freq_env = self.freq_env(num_samples, freq_decay)

        # Generate oscillators
        y1 = self.osc1(num_samples, osc1_freq, freq_env, osc1_mod)
        y2 = self.osc2(num_samples, osc2_freq, freq_env, osc2_mod)

        # Generate oscillator envelopee
        env1 = self.osc1_env(num_samples, osc1_decay)
        env2 = self.osc2_env(num_samples, osc2_decay)

        # Apply envelopes and sum oscillators
        y = self.osc1_gain(y1, osc1_gain) * env1
        y = y + self.osc2_gain(y2, osc2_gain) * env2

        # Generate noise
        noise = self.noise(params.shape[0], num_samples, device=params.device)
        noise_env = self.noise_env(num_samples, noise_decay)
        noise = noise * noise_env
        noise = self.noise_filter(noise, noise_freq, noise_q)
        noise = self.noise_gain(noise, noise_gain)

        # Add noise and waveshape
        y = y + noise
        y = self.tanh(y, tanh_gain)

        return y


class SimpleSynth(torch.nn.Module):
    """
    Basic synthesizer that generates a sine wave with a static envelope
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate

        # Static envelope
        self.decay_rate = 1.0 - (6.91 / sample_rate)
        self.env = torch.ones(sample_rate)
        self.env[1:] = self.decay_rate
        self.env = torch.cumprod(self.env, dim=0)

    def forward(self, gain: torch.Tensor):
        f0 = torch.ones(1, self.sample_rate) * 100.0
        f0 = 2 * torch.pi * f0 / self.sample_rate
        phase = torch.cumsum(f0, dim=-1)
        y = torch.sin(phase)
        y = y * self.env

        return y * gain
