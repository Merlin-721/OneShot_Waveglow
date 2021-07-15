python3 main.py \
 	-c config.yaml \
	-d preprocess/data/VCTK/ \
	-train_set train_128 \
	-train_index_file train_samples_128.json \
	-store_model_path models/vctk_22kHz/vctk_22_kl_1_full \
	-t vctk_22_kl_1_full \
	-iters 200000 \
	-summary_step 500