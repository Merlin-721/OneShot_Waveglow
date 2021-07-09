import pickle
from librosa import display
import matplotlib.pyplot as plt
import IPython.display as ipd
import random

def plot_data(data, instances, figsize=(16, 4)):
	fig, axes = plt.subplots(1, len(instances), figsize=figsize)
	for i, inst in enumerate(instances):
		axes[i].imshow(data[inst].T, aspect='auto', origin='lower', 
			interpolation='none')
	plt.show()
	
if __name__ == '__main__':

	path = "data/LJ/waveglow_form_mels/train_128.pkl"
	with open(path, "rb") as file:
		data = pickle.load(file)

	specs = list(data.keys())

	selected = []
	n_specs = 3
	for i in range(n_specs):
		selected.append(random.choice(specs))
	
	plot_data(data, tuple(selected))
	print()