import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys

def plot_data(data, title=None):
	fig, axes = plt.subplots(1,1)
	axes.imshow(data.T, aspect='auto', origin='lower', 
		interpolation='none', norm=Normalize(-18,2))
	if title is not None:
		plt.title(title)
	plt.show()

	
if __name__ == '__main__':
	path = sys.argv[1]

	with open(path, "rb") as file:
		spec = torch.load(file)

	plot_data(spec)