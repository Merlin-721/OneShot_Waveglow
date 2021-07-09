import torch
import matplotlib.pyplot as plt
import sys

def plot_data(data):
	fig, axes = plt.subplots(1,1)
	axes.imshow(data.T, aspect='auto', origin='lower', 
		interpolation='none')
	plt.show()
	
if __name__ == '__main__':

	path = sys.argv[1]

	with open(path, "rb") as file:
		spec = torch.load(file)

	plot_data(spec)