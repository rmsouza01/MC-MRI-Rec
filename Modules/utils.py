import numpy as np

def sum_of_squares(img_channels):
	"""
	Combines complex channels with square root sum of squares.
	"""
	sos =np.sqrt((np.abs(img_channels)**2).sum(axis = -1))
	return sos
	
