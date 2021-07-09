import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model import AE
from utils import *
from functools import reduce
import json
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from scipy.io.wavfile import write
import random
# from preprocess.tacotron.utils import get_spectrograms
from preprocess.tacotron2 import audio_processing
from preprocess.mel2samp import load_wav_to_torch, Mel2Samp
import librosa 

class OneShotInferencer(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)
        # args store other information
        self.args = args
        print(self.args)

        # init the model with config
        self.build_model()

        # load model
        self.load_model()

        with open(self.args.attr, 'rb') as f:
            self.attr = pickle.load(f)

    def load_model(self):
        print(f'Load model from {self.args.model}')
        self.model.load_state_dict(torch.load(f'{self.args.model}'))
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        print(self.model)
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
        dec = self.denormalize(dec)
        # this is where to integrate waveglow (ie replace griffin lim)
        # wav_data = melspectrogram2wav(dec)
        return dec

    def denormalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret

    # def write_wav_to_file(self, wav_data, output_path):
    #     write(output_path, rate=self.args.sample_rate, data=wav_data)
    #     return

    def write_mel_to_file(self, mel_data, output_path):
        print(f"Writing mel to {output_path}")
        torch.save(mel_data,output_path)

    def inference_from_path(self, waveglow_config):
        MelProcessor = Mel2Samp(**waveglow_config)
        src_audio, _ = load_wav_to_torch(self.args.source)
        tar_audio, _ = load_wav_to_torch(self.args.target)
        src_mel = np.array(MelProcessor.get_mel(src_audio).T)
        tar_mel = np.array(MelProcessor.get_mel(tar_audio).T)
        src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
        
        conv_mel = self.inference_one_utterance(src_mel, tar_mel)
        # self.write_wav_to_file(conv_wav, self.args.output)
        self.write_mel_to_file(conv_mel,self.args.output)
        return
        
# python inference.py -a attr.pkl -c config.yaml -model vctk_model.ckpt -s eg_wavs/p255_001.wav -t eg_wavs/p240_001.wav -o output_wavs/test4.wav -sr 24000
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-attr', '-a', help='attr file path')
    parser.add_argument('-config', '-c', help='config file path')
    parser.add_argument('-model', '-m', help='model path')
    parser.add_argument('-source', '-s', help='source wav path')
    parser.add_argument('-target', '-t', help='target wav path')
    parser.add_argument('-output', '-o', help='output wav path')
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)
    parser.add_argument('-data_config_path', '-w', help='path to config matching waveglow model')
    args = parser.parse_args()
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)

    with open(args.data_config_path) as f:
        waveglow_config = f.read()

    data_config = json.loads(waveglow_config)["data_config"]
    data_config["training_files"] = "preprocess/data/VCTK/22kHz_mels/train_files.txt"

    inferencer = OneShotInferencer(config=config, args=args)
    inferencer.inference_from_path(data_config)
