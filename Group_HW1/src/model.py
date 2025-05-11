import torch
import torch.nn as nn
import torchaudio

_SAMPLE_RATE = 16_000
_N_MELS = 80
_N_FFT = 400
_HOP_LENGTH = 160
_F_MIN = 50
_F_MAX = 7_600
_TOP_DB = 80

_mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=_SAMPLE_RATE, n_fft=_N_FFT, hop_length=_HOP_LENGTH, n_mels=_N_MELS, f_min=_F_MIN, f_max=_F_MAX
)
_to_db_transform = torchaudio.transforms.AmplitudeToDB(top_db=_TOP_DB)

def _waveform_to_fbank(wave_cpu):
    with torch.no_grad():
        mel_output = _mel_spec_transform(wave_cpu)
        fbank = _to_db_transform(mel_output)
    return fbank

class CharTokenizer:
    def __init__(self, idx2char_list):
        self.idx2char = idx2char_list

    def decode(self, idxs, collapse_repeats=False, remove_blanks=True):
        out, prev = [], None
        for i in idxs:
            if remove_blanks and i == 0:
                continue
            if collapse_repeats and prev == i:
                continue
            out.append(self.idx2char[i])
            prev = i
        return "".join(out)

class SeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel // 2) * dilation
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size=kernel, stride=stride,
            padding=padding, dilation=dilation, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)

class QuartzNetBlock(nn.Module):
    def __init__(self, ch, kernel, repeats, dropout):
        super().__init__()
        layers = []
        for _ in range(repeats):
            layers.append(SeparableConv1d(ch, ch, kernel, dropout=dropout))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x)

class QuartzNetSmallCTC(nn.Module):
    def __init__(self, vocab_size, in_feats=80, channels=384, dropout=0.2):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(in_feats, channels, kernel_size=11, padding=11//2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        blocks = []
        for _ in range(5): 
            blocks.append(QuartzNetBlock(channels, kernel=11, repeats=5, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        
        self.final = SeparableConv1d(channels, channels*2, kernel=29, dropout=dropout)
        self.classifier = nn.Conv1d(channels*2, vocab_size, kernel_size=1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, feats, feat_lens):
        x = feats.transpose(1, 2)
        x = self.input(x)
        x = self.blocks(x)
        x = self.final(x)
        logits = self.classifier(x)
        logp = self.log_softmax(logits.transpose(1, 2))
        return logp.transpose(0, 1)

def _greedy_decode_ctc(log_probs_ctc, tokenizer_instance):
    tokens = log_probs_ctc.argmax(-1).transpose(0, 1)
    return [
        tokenizer_instance.decode(seq.cpu().numpy(), collapse_repeats=True, remove_blanks=True) 
        for seq in tokens
    ]

class ASRModel:
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.idx2char = checkpoint['vocab']
        self.tokenizer = CharTokenizer(idx2char_list=self.idx2char)
        vocab_size = len(self.idx2char)
        
        saved_cfg = checkpoint.get('cfg', {})
        
        model_channels = 384
        model_dropout = saved_cfg.get('dropout', 0.2)
        
        self.model = QuartzNetSmallCTC(
            vocab_size=vocab_size,
            in_feats=_N_MELS, 
            channels=model_channels,
            dropout=model_dropout 
        ).to(self.device)
        
        model_state_key = 'model' if 'model' in checkpoint else 'model_state'
        self.model.load_state_dict(checkpoint[model_state_key])
        self.model.eval()

        self.resampler_cache = {}

    def _resample_waveform(self, wave_cpu, original_sr):
        if original_sr == _SAMPLE_RATE:
            return wave_cpu
        if original_sr not in self.resampler_cache:
            self.resampler_cache[original_sr] = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=_SAMPLE_RATE
            )
        return self.resampler_cache[original_sr](wave_cpu)

    def transcribe_waveform(self, waveform, original_sample_rate):
        wave_cpu = waveform.cpu()
        if wave_cpu.ndim == 1:
            wave_cpu = wave_cpu.unsqueeze(0)

        resampled_wave_cpu = self._resample_waveform(wave_cpu, original_sample_rate)
        
        fbank_feat_cpu = _waveform_to_fbank(resampled_wave_cpu)
        
        features_for_model = fbank_feat_cpu.squeeze(0).transpose(0, 1)
        
        feats_batched = features_for_model.unsqueeze(0).to(self.device)
        feat_lens = torch.tensor([features_for_model.shape[0]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            log_probs = self.model(feats_batched, feat_lens)
        
        predicted_text = _greedy_decode_ctc(log_probs, self.tokenizer)[0]
        return predicted_text

    def transcribe_file(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        return self.transcribe_waveform(waveform, sample_rate)