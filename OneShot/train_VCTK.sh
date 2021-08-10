python3 main.py \
 	-c config.yaml \
	-d preprocess/data/VCTK/ \
	-train_set train_128 \
	-train_index_file train_samples_128.json \
	-store_model_path models/vctk_22kHz_dropout/vctk_22_dropout \
	-t vctk_22_dropout \
	-iters 300000 \
	-summary_step 500