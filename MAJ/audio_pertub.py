import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, filtfilt


def _normalize(x):
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return x


def high_freq_rolloff(x, sr, cutoff=1500, order=4):
    nyquist = 0.5 * sr
    normalized_cutoff = min(max(cutoff / nyquist, 1e-6), 0.999999)
    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, x)


def add_breath_noise(x, sr, noise_level=0.005):
    noise = np.random.normal(0, noise_level, size=len(x))
    noise = high_freq_rolloff(noise, sr, cutoff=4000, order=2)
    return x + noise


def apply_reverb(input_audio, ir_audio, output_audio):
    x, sr = librosa.load(input_audio, sr=None)
    ir, _ = librosa.load(ir_audio, sr=sr)
    x_reverb = fftconvolve(x, ir, mode="full")
    x_reverb = _normalize(x_reverb)
    sf.write(output_audio, x_reverb, sr)


def add_echo(x, sr, delay=0.2, decay=0.5):
    delay_samples = int(sr * delay)
    echo_signal = np.zeros(len(x) + delay_samples)
    echo_signal[: len(x)] += x
    echo_signal[delay_samples:] += decay * x
    echo_signal = _normalize(echo_signal)
    return echo_signal


def apply_echo(input_audio, output_audio, delay=0.2, decay=0.5):
    x, sr = librosa.load(input_audio, sr=None)
    x_echo = add_echo(x, sr, delay=delay, decay=decay)
    sf.write(output_audio, x_echo, sr)


def simulate_whisper(input_audio, output_audio, reduction_factor=0.3):
    x, sr = librosa.load(input_audio, sr=None)
    x_soft = x * reduction_factor
    x_whisper = high_freq_rolloff(x_soft, sr, cutoff=1500, order=4)
    x_whisper = add_breath_noise(x_whisper, sr, noise_level=0.005)
    x_whisper = _normalize(x_whisper)
    sf.write(output_audio, x_whisper, sr)
