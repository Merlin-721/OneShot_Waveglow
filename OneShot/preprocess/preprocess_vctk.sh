. OneShot/preprocess/vctk.config

if [ $stage -le 0 ]; then
    printf "Making dataset with make_datasets_vctk.py\n"
    python3 OneShot/preprocess/make_datasets_vctk.py $raw_data_dir/wav22 $raw_data_dir/speaker-info.txt $data_dir $n_out_speakers $test_prop $waveglow_config
fi

if [ $stage -le 1 ]; then
    printf "\n Trimming spectograms with reduce_dataset.py\n"
    python3 OneShot/preprocess/reduce_dataset.py $data_dir/train.pkl $data_dir/train_$segment_size.pkl $segment_size
fi

if [ $stage -le 2 ]; then
    # sample training samples
    printf "\n Making json with sample_single_segments.py\n"
    python3 OneShot/preprocess/sample_single_segments.py $data_dir/train.pkl $data_dir/train_samples_$segment_size.json $training_samples $segment_size
fi
if [ $stage -le 3 ]; then
    # sample testing samples
    printf "\n Making jsons for in_test\n"
    python3 OneShot/preprocess/sample_single_segments.py $data_dir/in_test.pkl $data_dir/in_test_samples_$segment_size.json $testing_samples $segment_size
    printf "\n Making jsons for out_test\n"
    python3 OneShot/preprocess/sample_single_segments.py $data_dir/out_test.pkl $data_dir/out_test_samples_$segment_size.json $testing_samples $segment_size
fi

if [ $stage -le 4 ]; then
    printf "\n Getting attributes for normalisation/denormalisation\n"
    python3 OneShot/preprocess/get_attr.py $data_dir/train_$segment_size.pkl $data_dir/VCTK_attr.pkl
fi