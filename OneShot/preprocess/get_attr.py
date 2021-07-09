import pickle
import numpy as np
import sys



def attributes(instances):
	instances = np.array(instances)
	channels = instances[0].shape[1]	
	print(channels)
	all_data = np.concatenate(instances)
	mean = np.mean(all_data, axis=0)
	std = np.std(all_data, axis=0)
	return mean, std

if __name__ == '__main__':
	path = sys.argv[1]
	save_path = sys.argv[2]
	path = "preprocess/data/VCTK/22kHz_mels/train_128.pkl"
	save_path = "preprocess/data/VCTK/22kHz_mels/VCTK_attr.pkl"

	with open(path,"rb") as f:
		instance_dict = pickle.load(f)

	instance_vals = list(instance_dict.values())
	mean,std = attributes(instance_vals)
	attr_dict = {"mean" : mean, "std" : std}
	
	with open(save_path,"wb") as f:
		pickle.dump(attr_dict, f)