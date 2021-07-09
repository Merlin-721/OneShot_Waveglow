import torch
import matplotlib.pyplot as plt

def plot_data(data):
	fig, axes = plt.subplots(1,1)
	axes.imshow(data.T, aspect='auto', origin='lower', 
		interpolation='none')
	plt.show()
	
if __name__ == '__main__':

	path = "converted_mels/vctk_22_first.wav.pt"

	with open(path, "rb") as file:
		spec = torch.load(file)

	plot_data(spec)