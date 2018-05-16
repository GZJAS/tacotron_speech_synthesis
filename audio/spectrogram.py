"""Audio processing using spectrograms"""

import copy
import os

import librosa
import numpy as np
from scipy import signal

from Hparams import Hparams as hp


def _get_spectrograms(wavpath):
    """Computes normalized log melspectrograms and log magnitude spectrograms from wavfile

        Args:
            wavpath: Path to the wav file
            returns: A 2d array of shape (T, n_mels) (melspectrogram) and a 2d array of shape
            (T, 1+n_fft/2)(magnitude spectrogram)
    """
    # load the wav file
    y, sr = librosa.load(wavpath, sr=hp.sampling_rate)

    # Trim
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # STFT
    hop_length = int(hp.sampling_rate * hp.frame_shift)
    win_length = int(hp.sampling_rate * hp.frame_length)
    stft = librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hop_length, win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(stft)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sampling_rate, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, T)

    # convert from amplitude values to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db_level + hp.max_db_level) / hp.max_db_level, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db_level + hp.max_db_level) / hp.max_db_level, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mag, mel


def _invert_spectrogram(S):
    """Compute the ISTFT of a spectrogram

        Args:
            S: Spectrogram
    """
    hop_length = int(hp.sampling_rate * hp.frame_shift)
    win_length = int(hp.sampling_rate * hp.frame_length)

    return librosa.istft(S, hop_length, win_length=win_length, window="hann")


def _griffin_lim(S):
    """Griffin-lim algorithm to iteratively estimate phase given a spectrogram
    """
    X_best = copy.deepcopy(S)
    for i in range(hp.num_griffin_lim_iters):
        X_t = _invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = S * phase
    X_t = _invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def spectrogram2wav(S):
    """Invert a log magnitude spectrogram to wav
    """
    # transpose
    S = S.T

    # de-noramlize
    S = (np.clip(S, 0, 1) * hp.max_db_level) - hp.max_db_level + hp.ref_db_level

    # to amplitude
    S = np.power(10.0, S * 0.05)

    # wav reconstruction
    wav = _griffin_lim(S)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def load_spectrograms(wavpath):
    """Computes melspectrograms and log magnitude spectrograms and presents them in a form suitable for the tacotron
    model
    """
    wavname = os.path.basename(wavpath)

    # compute the spectrograms
    mag, mel = _get_spectrograms(wavpath)

    # pad and reshape the spectrograms
    num_frames = mel.shape[0]
    num_paddings = hp.r - (num_frames % hp.r) if num_frames % hp.r != 0 else 0

    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")

    return wavname, mel.reshape((-1, hp.n_mels * hp.r)), mag
