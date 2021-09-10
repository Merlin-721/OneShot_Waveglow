import torch
import numpy as np
import torch.nn.functional as F
import yaml
import pickle
from .model import AE
from .utils import *
import json
from argparse import ArgumentParser
from Waveglow.mel2samp import load_wav_to_torch, Mel2Samp
from show_mel import plot_data
import time

class OneShotInferencer(object):
    def __init__(self, config, args, waveglow_config, verbose=True):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        # args store other information
        self.args = args
        if verbose:
            print(self.args)
            print(config)
            print(self.model)

        # init the model with config
        self.MelProcessor = Mel2Samp(**waveglow_config)
        self.build_model()

        # load model
        self.load_model()

        with open(self.args.attr, 'rb') as f:
            self.attr = pickle.load(f)

    def load_model(self):
        print(f'Load model from {self.args.oneshot_model}')
        self.model.load_state_dict(torch.load(f'{self.args.oneshot_model}'))
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        self.model.eval()
        return

    def utt_make_frames(self, x):
        frame_size = self.config['data_loader']['frame_size']
        remains = x.size(0) % frame_size 
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out

    def inference_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)
        x_cond = self.utt_make_frames(x_cond)
        dec = self.model.inference(x, x_cond)
        dec = dec.transpose(1, 2).squeeze(0)
        dec = dec.detach().cpu().numpy()
        return dec


    def remove_noise(self,mel,tar_mel, strength=0.2, mode='zeros'):
        minval = np.min(mel)
        if mode == 'blank':
            denoise_mel = torch.full(mel.shape, minval, dtype=torch.float32).cuda()
            # denoise_mel = torch.zeros(mel.shape).cuda()
        elif mode == 'rand':
            denoise_mel = torch.randn(mel.shape).cuda()
        else:
            raise Exception("Denoise mode must be blank or rand")
        noise_mel = self.inference_one_utterance(denoise_mel, tar_mel)
        noise_mel = np.mean(noise_mel,axis=0,keepdims=True)

        noise_removed = mel - (noise_mel * strength)
        return noise_removed

    def write_mel_to_file(self, mel_data, output_path):
        print(f"Writing mel to {output_path}")
        torch.save(mel_data,output_path)

    def inference(self, src_audio, tar_audio, plot=False):
        src_mel = torch.from_numpy(np.array(self.MelProcessor.get_mel(src_audio).T)).cuda()
        tar_mel = torch.from_numpy(np.array(self.MelProcessor.get_mel(tar_audio).T)).cuda()
        if plot:
            plot_data(src_mel,save_loc=f"output_mels/SourceMel.png",show=False)
            plot_data(tar_mel,save_loc=f"output_mels/TargetMel.png",show=False)

        start_time = time.time()
        conv_mel = self.inference_one_utterance(src_mel, tar_mel)
        duration = time.time() - start_time
        conv_mel = self.remove_noise(conv_mel, tar_mel, strength=0.1, mode='blank')
        if plot:
            plot_data(conv_mel,save_loc=f"output_mels/ConvertedMel.png",show=False)
        return conv_mel, duration

    def inference_from_audio(self, src_audio, target, plot=False):
        """
        Source audio is array
        Target is path
        """
        src_audio = torch.tensor(src_audio)
        tar_audio,_ = load_wav_to_torch(target)
        conv_mel, duration = self.inference(src_audio, tar_audio, plot=plot)
        return conv_mel, duration
        
    def inference_from_path(self, source, target, plot=False):
        src_audio,_ = load_wav_to_torch(source)
        tar_audio,_ = load_wav_to_torch(target)
        conv_mel, duration = self.inference(src_audio, tar_audio, plot=plot)
        return conv_mel, duration


