python dual_inference.py \
	-source input_wavs/p336_021.wav \
	-target input_wavs/p284_024.wav \
	-output output_wavs/22_dropout/sig_0.6_den_0.1 \
	-output_name nodrop\
	-attr OneShot/preprocess/data/VCTK/VCTK_attr.pkl \
	-oneshot_conf OneShot/models/vctk_22kHz_dropout/vctk_22_dropout.config.yaml \
	-oneshot_model OneShot/models/vctk_22_dropout_100k.ckpt\
	-sample_rate 22050 \
	-data_conf Waveglow/config.json \
	-waveglow_path Waveglow/models/waveglow_256channels_universal_v5.pt \
	-sigma 1\
	-denoiser_strength 0.1
