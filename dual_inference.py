import torch
import yaml
import json
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
	
	oneshot_inferencer = OneShotInferencer(config=oneshot_conf, args=args)
	waveglow_inferencer = WaveglowInferencer(args)

	with open(args.data_conf) as f:
		data_config = f.read()
	data_config = json.loads(data_config)["data_config"]

	print("\nRunning OneShot")
	mel = oneshot_inferencer.inference_from_path(data_config)
	# plot_data(mel)
	print("\nRunning Waveglow")
	# name = f"{args.oneshot_model.split('/')[-1][-9:-5]}_sig_{args.sigma}_den_{args.denoiser_strength}_{args.output_name}"
	name = f"{args.oneshot_model.split('/')[-1][-9:-5]}_{args.source.split('/')[-1][:4]}_{args.target.split('/')[-1][:4]}_{args.output_name}"
	waveglow_inferencer.inference(mel.T, name)