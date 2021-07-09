python inference.py \
 -a preprocess/data/VCTK/22kHz_mels/VCTK_attr.pkl \
 -c models/vctk_22kHz/vctk_22kHz_model.config.yaml \
 -m models/vctk_22kHz/vctk_22kHz_model.ckpt \
 -s eg_wavs/22k_1.wav \
 -t eg_wavs/22k_2.wav \
 -o converted_mels/22k_test.wav.pt \
 -w preprocess/waveglow_VCTK_config.json \
 -sample_rate 22050

python show_mel.py \
	converted_mels/22k_test.wav.pt