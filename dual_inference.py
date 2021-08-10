import torch
import numpy as np
import yaml
import json
from pathlib import Path
import os
from argparse import ArgumentParser
from OneShot.show_mel import plot_data
from OneShot.inference import OneShotInferencer
from Waveglow.inference import WaveglowInferencer

if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('-source', '-s', help='source wav path')
	parser.add_argument('-target', '-t', help='target wav path')
	parser.add_argument('-output', '-o', help='output wav path')
	parser.add_argument('-output_name', help='name of output file')

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

	# if source is directory, convert all files	
	if args.source[-1] == '/':
		wav_fpaths = list(Path(args.source).glob("**/*.wav"))
		speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))	
	else:
		wav_fpaths = [Path(args.source)]
		speakers = ["output"]
	target = Path(args.target)

	oneshot_inferencer = OneShotInferencer(config=oneshot_conf, args=args)
	waveglow_inferencer = WaveglowInferencer(args)

	with open(args.data_conf) as f:
		data_config = f.read()
	data_config = json.loads(data_config)["data_config"]

	for i, speaker in enumerate(np.unique(speakers)):	
		utts = np.where(np.array(speakers) == speaker)

		if not os.path.exists(f"{args.output}/{speaker}"):
			os.makedirs(f"{args.output}/{speaker}")

		for source in np.array(wav_fpaths)[utts]:

			print("\nRunning OneShot")
			mel = oneshot_inferencer.inference_from_path(data_config, source, target)

			print("\nRunning Waveglow")
			name = f"{speaker}/{source.name.split('.')[0]}_{target.name.split('.')[0]}"
			waveglow_inferencer.inference(mel.T, name)