
python -W ignore AutoConvert.py \
	-target=Presentation_Target/p269_023.wav \
	-attr=OneShot/models/vctk_colab/VCTK_attr.pkl \
	-oneshot_conf=OneShot/models/vctk_colab/.config.yaml \
	-oneshot_model=OneShot/models/vctk_22_kl_1_full_200k.ckpt \
	-sample_rate=22050 \
	-data_conf=Waveglow/config.json \
	-waveglow_path=Waveglow/models/waveglow_256channels_universal_v5.pt \
	-sigma=1\
	-denoiser_strength=0
