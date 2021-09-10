import yaml
import json
from pathlib import Path
import os
from argparse import ArgumentParser
from OneShot.inference import OneShotInferencer
from Waveglow.inference import WaveglowInferencer
import sounddevice as sd
from record_anylength import record_audio
from time import sleep

if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('-target', '-t', help='target wav path')

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

	target = Path(args.target)
	if not os.path.exists(target):
		raise Exception(f"Target file {target} does not exist")

	oneshot_inferencer = OneShotInferencer(config=oneshot_conf, args=args, verbose=False)
	waveglow_inferencer = WaveglowInferencer(args)

	with open(args.data_conf) as f:
		data_config = f.read()
	data_config = json.loads(data_config)["data_config"]

	source = Path("wav_tmp/input.wav")
	while True: 
		while True:
			try:
				sleep(2)
				print("Waiting...")
			except KeyboardInterrupt:
				break
		record_audio(source, args.sample_rate)
		mel,_ = oneshot_inferencer.inference_from_path(data_config, source, target)

		audio = waveglow_inferencer.inference(mel.T, None, save_wav=False)
		sd.play(audio, args.sample_rate)
		sd.wait()

		os.remove(source)
