import torch
import torch.nn as nn
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram


class Wav2Mel(nn.Module):
    """Transform audio file into mel spectrogram tensors."""

    def __init__(
        self,
        sample_rate: int = 16000,
        norm_db: float = -3.0,
        sil_threshold: float = 1.0,
        sil_duration: float = 0.1,
        fft_window_ms: float = 25.0,
        fft_hop_ms: float = 10.0,
        f_min: float = 50.0,
        n_mels: int = 80,
    ):
        """
        Initialize the Wav2Mel module.

        Args:
            sample_rate (int): The sample rate of the input waveform.
            norm_db (float): The normalization level in dB.
            sil_threshold (float): The silence threshold in percentage.
            sil_duration (float): The minimum silence duration in seconds.
            fft_window_ms (float): The duration of the FFT window in milliseconds.
            fft_hop_ms (float): The duration of the FFT hop in milliseconds.
            f_min (float): The minimum frequency of the mel filterbank.
            n_mels (int): The number of mel filterbanks.
        """
        super().__init__()
        # Initialize the SoxEffects and LogMelspectrogram modules
        self.sample_rate = sample_rate
        self.norm_db = norm_db
        self.sil_threshold = sil_threshold
        self.sil_duration = sil_duration
        self.fft_window_ms = fft_window_ms
        self.fft_hop_ms = fft_hop_ms
        self.f_min = f_min
        self.n_mels = n_mels

        self.sox_effects = SoxEffects(sample_rate, norm_db, sil_threshold, sil_duration)
        self.log_melspectrogram = LogMelspectrogram(
            sample_rate, fft_window_ms, fft_hop_ms, f_min, n_mels
        )

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Transform audio files into log mel spectrogram tensors.

        Args:
            wav_tensor (torch.Tensor): The input waveform tensor.
            sample_rate (int): The sample rate of the input waveform.

        Returns:
            torch.Tensor: The log mel spectrogram tensor.
        """
        # Apply SoX effects to the waveform tensor
        wav_tensor = self.sox_effects(wav_tensor, sample_rate)
        # Compute the log mel spectrogram from the processed waveform tensor
        mel_tensor = self.log_melspectrogram(wav_tensor)
        return mel_tensor


class SoxEffects(nn.Module):
    """Transform waveform tensors."""

    def __init__(
        self,
        sample_rate: int,
        norm_db: float,
        sil_threshold: float,
        sil_duration: float,
    ):
        """
        Initialize the SoxEffects module.

        Args:
            sample_rate (int): The sample rate of the input waveform.
            norm_db (float): The normalization level in dB.
            sil_threshold (float): The silence threshold in percentage.
            sil_duration (float): The minimum silence duration in seconds.
        """
        super().__init__()
        # Define the sequence of SoX effects to be applied
        self.effects = [
            ["channels", "1"],  # convert to mono
            ["rate", f"{sample_rate}"],  # resample
            ["norm", f"{norm_db}"],  # normalize to -3 dB
            [
                "silence",
                "1",
                f"{sil_duration}",
                f"{sil_threshold}%",
                "-1",
                f"{sil_duration}",
                f"{sil_threshold}%",
            ],  # remove silence throughout the file
        ]

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Apply SoX effects to the input waveform tensor.

        Args:
            wav_tensor (torch.Tensor): The input waveform tensor.
            sample_rate (int): The sample rate of the input waveform.

        Returns:
            torch.Tensor: The transformed waveform tensor.
        """
        wav_tensor, _ = apply_effects_tensor(wav_tensor, sample_rate, self.effects)
        return wav_tensor


class LogMelspectrogram(nn.Module):
    """Transform waveform tensors into log mel spectrogram tensors."""

    def __init__(
        self,
        sample_rate: int,
        fft_window_ms: float,
        fft_hop_ms: float,
        f_min: float,
        n_mels: int,
    ):
        """
        Initialize the LogMelspectrogram module.

        Args:
            sample_rate (int): The sample rate of the input waveform.
            fft_window_ms (float): The duration of the FFT window in milliseconds.
            fft_hop_ms (float): The duration of the FFT hop in milliseconds.
            f_min (float): The minimum frequency of the mel filterbank.
            n_mels (int): The number of mel filterbanks.
        """
        super().__init__()
        # Initialize the MelSpectrogram module
        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=int(sample_rate * fft_hop_ms / 1000),
            n_fft=int(sample_rate * fft_window_ms / 1000),
            f_min=f_min,
            n_mels=n_mels,
        )

    def forward(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the log mel spectrogram from the input waveform tensor.

        Args:
            wav_tensor (torch.Tensor): The input waveform tensor.

        Returns:
            torch.Tensor: The log mel spectrogram tensor.
        """
        # Compute the mel spectrogram and apply log transformation
        mel_tensor = self.melspectrogram(wav_tensor).squeeze(0).T  # (time, n_mels)
        return torch.log(torch.clamp(mel_tensor, min=1e-9))
