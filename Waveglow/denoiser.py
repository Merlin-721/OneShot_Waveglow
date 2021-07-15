import sys
sys.path.append('tacotron2')
import torch
from tacotron2.layers import TacotronSTFT
from Waveglow.mel2samp import MAX_WAV_VALUE

def get_mel(audio,stft):
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec

class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, waveglow, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros'):
        super(Denoiser, self).__init__()
        self.stft = TacotronSTFT(filter_length=filter_length,
                         hop_length=int(256),
                         win_length=win_length).cuda()
        if mode == 'zeros':
            mel_input = torch.zeros(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype,
                device=waveglow.upsample.weight.device)
        elif mode == 'normal':
            mel_input = torch.randn(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype,
                device=waveglow.upsample.weight.device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = waveglow.infer(mel_input, sigma=0.0).float()
            self.bias_spec = get_mel(audio=bias_audio,stft=self.stft)
            # self.bias_spec = self.stft.mel_spectrogram(bias_audio)

        # self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self,mel, strength=0.1):
        noise = torch.mean(self.bias_spec, axis=1)
        noise = torch.clamp(noise, max=0.0)
        audio_spec_denoised = mel.T - (noise * strength)
        return audio_spec_denoised.T
