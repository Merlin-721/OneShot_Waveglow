import pickle
import sys
import glob 
import random
import os
from collections import defaultdict
import re
import json
import numpy as np
from Waveglow.mel2samp import load_wav_to_torch, Mel2Samp
from itertools import product


def read_speaker_info(speaker_info_path):
    speaker_ids = []
    with open(speaker_info_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            speaker_id = line.strip().split()[0]
            speaker_ids.append(speaker_id)
    return speaker_ids

def read_filenames(root_dir):
    speaker2filenames = defaultdict(lambda : [])
    for path in sorted(glob.glob(os.path.join(root_dir, '*/*'))):
        filename = path.strip().split('/')[-1]
        speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
        speaker2filenames[speaker_id].append(path)
    return speaker2filenames

def write_file_list(output_dir, path_list, list_name):
    with open(os.path.join(output_dir, list_name), 'w') as f:
        for path in path_list:
            f.write(f'{path}\n')


def main(data_dir, speaker_info_path, output_dir, test_speakers, test_proportion, waveglow_config):

    MelProcessor = Mel2Samp(**waveglow_config)

    speaker_ids = read_speaker_info(speaker_info_path)
    random.shuffle(speaker_ids)

    train_speaker_ids = speaker_ids[:-test_speakers]
    test_speaker_ids = speaker_ids[-test_speakers:]

    speaker2filenames = read_filenames(data_dir)

    train_path_list, in_test_path_list, out_test_path_list = [], [], []

    for speaker in train_speaker_ids:
        path_list = speaker2filenames[speaker]
        random.shuffle(path_list)
        test_data_size = int(len(path_list) * test_proportion)
        train_path_list += path_list[:-test_data_size]
        in_test_path_list += path_list[-test_data_size:]

    
    write_file_list(output_dir, train_path_list,'train_files.txt')
    write_file_list(output_dir, in_test_path_list,'in_test_files.txt')

    for speaker in test_speaker_ids:
        path_list = speaker2filenames[speaker]
        out_test_path_list += path_list

    write_file_list(output_dir, out_test_path_list,'out_test_files.txt')


    for dset, path_list in zip(['train', 'in_test', 'out_test'], \
            [train_path_list, in_test_path_list, out_test_path_list]):
        total_convert = len(path_list)
        print(f'processing {dset} set, {total_convert} files')
        data = {}
        for i, path in enumerate(sorted(path_list)):
            if i % 500 == 0:
                print(f"Converted {i} of {total_convert}")
            # ACTUAL CONVERSION HERE
            audio, _ = load_wav_to_torch(path)
            melspectrogram = MelProcessor.get_mel(audio)

            filename = path.strip().split('/')[-1]
            data[filename] = np.array(melspectrogram.T)

        output_path = os.path.join(output_dir, f'{dset}.pkl')
        print(f"Saving {dset} as {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)




if __name__ == '__main__':
    data_dir = sys.argv[1] 
    speaker_info_path = sys.argv[2] 
    output_dir = sys.argv[3] 
    test_speakers = int(sys.argv[4])
    test_proportion = float(sys.argv[5])
    waveglow_config = sys.argv[6] 

    with open(waveglow_config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]

    main(data_dir, speaker_info_path, output_dir, test_speakers, test_proportion, data_config)