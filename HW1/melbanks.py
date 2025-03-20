from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
 
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power
        self.f_min_hz = f_min_hz

        self.f_min_hz = f_min_hz
        self.f_max_hz = samplerate / 2 if f_max_hz is None else f_max_hz
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        n_freqs = self.n_fft // 2 + 1 if self.onesided else self.n_fft
        return F.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale
        )

    def spectrogram(self, x):
        return torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex
        )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        spec_complex = self.spectrogram(x)
        if spec_complex.is_complex():
            spec_power = torch.abs(spec_complex) ** self.power
        else:
            spec_power = (spec_complex[..., 0] ** 2 + spec_complex[..., 1] ** 2) ** (self.power / 2)
        
        mel_fbanks_t = self.mel_fbanks.T.to(x.device)

        mel_spec = torch.matmul(mel_fbanks_t, spec_power)

        log_mel_spec = torch.log(mel_spec + 1e-6)
        
        return log_mel_spec