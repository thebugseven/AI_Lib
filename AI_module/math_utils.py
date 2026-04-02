import numpy as np

def softmax(z):
	z = np.array(z)
	z_max = np.max(z)
	exp_vals = np.exp(z - z_max)
	s = np.sum(exp_vals)
	return exp_vals / s