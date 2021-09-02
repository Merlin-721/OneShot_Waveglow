import torch
import numpy as np
import yaml
import json
from pathlib import Path
import os
from argparse import ArgumentParser
from show_mel import plot_data
from OneShot.inference import OneShotInferencer
from Waveglow.inference import WaveglowInferencer
import time

def files_from_path(path):
	if os.path.isdir(path):
		wav_fpaths = list(Path(path).glob('**/*.wav'))
		speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))	
	else:
		wav_fpaths = [Path(path)]
		speakers = [Path(path).stem]
	return wav_fpaths, speakers

if __name__ == '__main__':
	parser = ArgumentParser()

	# Source can be single file or directory of speakers
	# Outputs to seperate directories 

	parser.add_argument('-source', '-s', help='source wav path')
	parser.add_argument('-target', '-t', help='target wav path')
	parser.add_argument('-output_dir', '-o', help='output wav path')

	# OneShot
	parser.add_argument('-attr', '-a', help='data mean & std attr file path')
	parser.add_argument('-oneshot_conf', '-c', help='OneShot config file path')
	parser.add_argument('-oneshot_model', '-m', help='OneShot model path')
	parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)
	parser.add_argument('-data_conf', help='data configuration json') # VCTK_config.json

	# Waveglow
	parser.add_argument('-waveglow_path','-w',  help='Path to waveglow decoder model')
	parser.add_argument("-sigma", default=1.0, type=float)
	parser.add_argument("--is_fp16", action="store_true")
	parser.add_argument("-denoiser_strength","-d", default=0.0, type=float, 
				help='Removes model bias. Start with 0.1 and adjust')


	args = parser.parse_args()

	with open(args.oneshot_conf) as f:
		oneshot_conf = yaml.load(f)

	wav_fpaths, speakers = files_from_path(args.source)
	target, _ = files_from_path(args.target)
	if len(target) != 1:
		raise Exception("Target folder must contain 1 wav file")
	target = target[0]

	if not os.path.exists(target):
		raise Exception(f"Target file {target} does not exist")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)


	oneshot_inferencer = OneShotInferencer(config=oneshot_conf, args=args)
	waveglow_inferencer = WaveglowInferencer(args)

	with open(args.data_conf) as f:
		data_config = f.read()
	data_config = json.loads(data_config)["data_config"]

	start = time.time()
	utt_count = 0
	for i, speaker in enumerate(np.unique(speakers)):	
		utts = np.where(np.array(speakers) == speaker)

		if not os.path.exists(f"{args.output_dir}/{speaker}"):
			os.makedirs(f"{args.output_dir}/{speaker}")

		for source in np.array(wav_fpaths)[utts]:
			utt_count += 1
			print(f"\nConverting {source.name} to {target.name}")
			print("Running OneShot")
			mel,_ = oneshot_inferencer.inference_from_path(data_config, source, target)

			print("Running Waveglow")
			name = f"{source.stem}_{target.stem}.wav"	
			out_path = Path(args.output_dir,speaker,name)	
			waveglow_inferencer.inference(mel.T, out_path)
	end = time.time()
	ET = end - start
	print(f"\nTotal time for {i+1} speakers and {utt_count} utterances: {ET}")