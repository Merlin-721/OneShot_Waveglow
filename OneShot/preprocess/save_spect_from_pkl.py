import pickle
import random
import torch
if __name__ == '__main__':
	path = "data/vctk/waveglow_form_mels/train_128.pkl"
	with open(path, "rb") as file:
		data = pickle.load(file)

	specs = list(data.keys())

	selected = []
	n_specs = 3
	for i in range(n_specs):
		selected.append(random.choice(specs))

	for sel in selected:
		torch.save(torch.tensor(data[sel]), f"waveglow_specs/{sel}.pt")