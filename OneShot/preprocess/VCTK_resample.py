import librosa
import soundfile
from multiprocessing import Pool
import glob
import os
from pathlib import Path

SR = 22050
ROOT_PATH = '/home/merlin/OneDrive/modules/individualProject/voiceChanger/Datasets/DS_10283_2119/VCTK-Corpus/wav48'
NEW_ROOT_PATH ='/home/merlin/OneDrive/modules/individualProject/voiceChanger/Datasets/VCTK_22050Hz'

def read_filenames(root_dir):
    path_list = []
    for path in sorted(glob.glob(os.path.join(root_dir, '*/*'))):
        path_list.append(path)
    return path_list

def downsample(path):

	speaker, filename = path.strip().split("/")[-2:]

	speaker_dir = f"{NEW_ROOT_PATH}/{speaker}"

	Path(speaker_dir).mkdir(parents=True, exist_ok=True)

	new_path = f"{speaker_dir}/{filename}"

	y, s = librosa.load(path, sr=SR) # Downsample 
	soundfile.write(new_path, y, SR)


if __name__ == '__main__':

	path_list = read_filenames(ROOT_PATH)

	with Pool(4) as p:
		p.map(downsample,path_list)