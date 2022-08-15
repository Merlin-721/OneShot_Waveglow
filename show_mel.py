import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys

def plot_data(data, title=None, norm=False, save_loc=None, show=True):
	if type(data) == torch.Tensor:
		data = data.cpu().numpy()
	fig, axes = plt.subplots(1,1)

	if norm:
		axes.imshow(data.T, aspect='equal', origin='lower', 
			interpolation='none', norm=Normalize(-12,2))
	else:
		axes.imshow(data.T, aspect='equal', origin='lower') #, 

	if title is not None:
		plt.title(title)
	if save_loc:
		plt.savefig(save_loc, dpi=300)
	if show:
		plt.show()

	
if __name__ == '__main__':
	path = sys.argv[1]

	with open(path, "rb") as file:
		spec = torch.load(file)

	plot_data(spec)