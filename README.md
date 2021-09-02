# References
The code contained in this work is an adaptation of:
https://github.com/jjery2243542/adaptive_voice_conversion and
https://github.com/NVIDIA/WaveGlow

License details for each can be found in OneShot/LICENSE and 
Waveglow/LICENSE respectively. 

Details of changes are outlined in the report:
"Vocal Style-Transfer for Anti-Bias Interviews", Section 3.3

Line-by-line changes are contained in the Differences directory.

I verify that I am the sole author of the programmes contained in
this archive, except where explicitly stated to the contrary.
Merlin Lindsay, 02.09.2021
## To Run Conversion
In terminal type:

	chmod +x run_dual_inference.sh 
	run_dual_inference.sh

Alternatively:

	python dual_inference.py \
		-source=input_wavs/sources/
		-target=input_wavs/target/target.wav
		-output_dir=output_wavs/ \
		-attr=OneShot/models/vctk_colab/VCTK_attr.pkl \
		-oneshot_conf=OneShot/models/vctk_colab/.config.yaml \
		-oneshot_model=OneShot/models/vctk_22_kl_1_full_200k.ckpt \
		-sample_rate=22050 \
		-data_conf=Waveglow/config.json \
		-waveglow_path=Waveglow/models/waveglow_256channels_universal_v5.pt \
		-sigma=1\
		-denoiser_strength=0.1


<!-- WaveGlow -->
### Dependencies 
* numpy
* torch==1.0
* tensorflow
* numpy==1.13.3
* inflect==0.2.5
* librosa==0.6.0
* scipy==1.0.0
* tensorflow==2.5.0 
* tensorboardX==1.1
* Unidecode==1.0.22
* tqdm
* pillow

# Global

| File | Description |
|----------------|----------------|
| dual_inference.py | Main script to perform audio-to-audio conversion. <li>Obtains paths to convert and creates target dir.</li><li>Loads configuration files.</li><li>Instantiates OneShot and WaveGlow inference models.</li><li>Performs conversion on each source wav, using the style of the target.</li><li>Saves converted wavs to output directory. </li> |
| run_dual_inference.sh | Shell script to call dual_inference.py. |



# OneShot


| File | Description |
|----------------|----------------|
| inference.py | Contains inferencer object which runs the forward pass through OneShot. <li> Loads the saved models</li><li> Generates spectrograms with WaveGlow's Mel2Samp object</li><li> Creates embedding of the target voice</li><li> Performs conversion, returning converted mel-spectrogram</li> |
| LICENSE | MIT License provided by creators of OneShot. |
| main.py | Generates data loader and training objects to run OneShot training loop. |
| model.py | Contains OneShot models. |
| utils.py | Logger for Tensorboard. |
| data_utils.py | Classes for creating batches for training. |
| config.yaml | Config file for OneShot. |
| solver.py | Training loop for OneShot model. |
| preprocess/ | Directory of files for generating train, test and validation sets. These can be run by executing preprocess_vctk.sh |
| models/ | Directory containing trained OneShot models. |



# WaveGlow

### Pre-trained WaveGlow vocoder:

| File | Description |
|----------------|----------------|
| inference.py | WaveGlow inferencer object, receiving mels from OneShot. <li>Denoises spectrogram</li><li>Runs inference</li><li>Converts to audio</li><li>Saves .wav</li> |
| mel2samp.py | Creates mel-spectrograms using Tacotron-2's STFT. This object is used to create training data, as well as extracting mel-specs during inference.|
| glow.py | Contains WaveGlow model, with loss function, and WaveNet affine coupling layers. |
| denoiser.py | Denoiser model for removing bias of WaveGlow from generated spectrogram |
| config.json | Configuration file of WaveGlow, specifying training, data and hyperparameter configs. |
| make_train_files.py | Creates two text files containing paths to wav files, split into training and test sets. |
| train.py | Training loop for WaveGlow. |
| distributed.py | Training loop for distributed training of WaveGlow across multiple GPUs.| 
| models/ | Directory containing pre-trained WaveGlow model. Pre-trained vocoder can be found at: https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view|
| tacotron2/ | Directory containing functions and objects from Tacotron-2 for mel-spectrogram generation. |
| LICENSE | BSD 3-Clause License. Copyright statement from Nvidia. |