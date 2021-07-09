python dual_inference.py \
	-source input_wavs/22k_1.wav \
	-target input_wavs/22k_2.wav \
	-output output_wavs/ \
	-attr OneShot/preprocess/data/VCTK_attr.pkl \
	-oneshot_conf OneShot/config.yaml\
	-oneshot_model OneShot/models/vctk_22kHz_model.ckpt\
	-sample_rate 22050 \
	-data_conf Waveglow/config.json \
	-waveglow_path Waveglow/models/waveglow_256channels_universal_v5.pt \
	-sigma 0.6\
	--is_fp16 \
	-denoiser_strength 0 \