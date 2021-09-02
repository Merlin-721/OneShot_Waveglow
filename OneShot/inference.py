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
        print(f'Load model from {self.args.oneshot_model}')
        self.model.load_state_dict(torch.load(f'{self.args.oneshot_model}'))
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
        return dec

    def denormalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret

    def remove_noise(self,mel,tar_mel, strength=0.2, mode='zeros'):
        if mode == 'blank':
            denoise_mel = torch.full(mel.shape, -12.0, dtype=torch.float32).cuda()
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

    def inference_from_path(self, waveglow_config, source, target):
        MelProcessor = Mel2Samp(**waveglow_config)
        src_audio, _ = load_wav_to_torch(source)
        tar_audio, _ = load_wav_to_torch(target)
        src_mel = np.array(MelProcessor.get_mel(src_audio).T)
        tar_mel = np.array(MelProcessor.get_mel(tar_audio).T)

        src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
        start_time = time.time()
        conv_mel = self.inference_one_utterance(src_mel, tar_mel)
        duration = time.time() - start_time
        conv_mel = self.remove_noise(conv_mel, tar_mel, strength=0.1, mode='blank')
        conv_mel = self.denormalize(conv_mel)
        return conv_mel, duration
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-source', '-s', help='source wav path')
    parser.add_argument('-target', '-t', help='target wav path')
    parser.add_argument('-output', '-o', help='output wav path')
    parser.add_argument('-attr', '-a', help='attr file path')
    parser.add_argument('-oneshot_conf', '-c', help='config file path')
    parser.add_argument('-oneshot_model', '-m', help='model path')
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)
    parser.add_argument('-data_config', '-w', help='path to config matching waveglow model')
    args = parser.parse_args()
    # load config file 
    with open(args.oneshot_conf) as f:
        oneshot_conf = yaml.load(f)

    with open(args.data_config_path) as f:
        waveglow_config = f.read()

    data_config = json.loads(waveglow_config)["data_config"]
    data_config["training_files"] = "preprocess/data/VCTK/22kHz_mels/train_files.txt"

    inferencer = OneShotInferencer(config=oneshot_conf, args=args)
    inferencer.inference_from_path(data_config,args.source,args.target)
