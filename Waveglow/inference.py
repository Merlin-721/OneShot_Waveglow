# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
from .mel2samp import files_to_list, MAX_WAV_VALUE
from .denoiser import Denoiser
from OneShot.show_mel import plot_data

class WaveglowInferencer(object):
    def __init__(self, args):
        self.args = args

        self.waveglow = torch.load(args.waveglow_path)['model']
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow.cuda().eval()

        if args.is_fp16:
            from apex import amp
            self.waveglow, _ = amp.initialize(self.waveglow, [], opt_level="O3")

        if args.denoiser_strength > 0:
            self.denoiser = Denoiser(self.waveglow).cuda()
        
    def inference(self, mel, filename, plot=False):
        # takes transposed mel from oneshot
        with torch.no_grad():
            mel = torch.from_numpy(mel.astype("float32")).unsqueeze(0).cuda()
            mel = torch.autograd.Variable(mel)

            if self.args.denoiser_strength > 0:
                mel = self.denoiser.forward(mel.squeeze(), self.args.denoiser_strength)
                mel = mel.unsqueeze(0)
            mel = mel.half() if self.args.is_fp16 else mel

            if plot:
                plot_data(mel.squeeze(0).cpu().numpy().T.astype("float32"), "WG Denoised mel")

            audio = self.waveglow.infer(mel, sigma=self.args.sigma)

            audio = audio * MAX_WAV_VALUE

            audio = audio.squeeze(0).cpu().numpy()
            audio = audio.astype("int16")
            audio_path = os.path.join(self.args.output_dir, 
                        "{}.wav".format(filename))

            print(f"Writing audio to {audio_path}")
            write(audio_path, self.args.sample_rate, audio)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-output_name", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    inferencer = WaveglowInferencer(args)

    files = files_to_list(args.filelist_path)
    for file in files:
        mel = torch.load(file)
        inferencer.inference(mel.T, args.output_name)
