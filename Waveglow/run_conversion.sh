python inference.py \
-f <(ls input_specs/*.pt) \
-w waveglow_256channels_universal_v5.pt \
-o output_wavs/ \
--is_fp16 \
-s 0.6 \
--sampling_rate 22050