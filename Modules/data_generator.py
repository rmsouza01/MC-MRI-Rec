import numpy as np
import h5py
import os
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, dim, under_masks, crop, batch_size, n_channels,nslices = 256, shuffle=True):
		self.list_IDs = list_IDs
		self.dim = dim
		self.under_masks = under_masks
		self.crop = crop # Remove slices with no or little anatomy
		self.batch_size = batch_size
		self.n_channels = n_channels
		self.nslices = nslices
		self.shuffle = shuffle
		self.nsamples = len(self.list_IDs)*(self.nslices - self.crop[0] - self.crop[1])
		self.on_epoch_end()
		
	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(self.nsamples/ self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'

	# Generate indexes of the batch
		batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, Y = self.__data_generation(batch_indexes)

		return X, Y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(self.nsamples)
		if self.shuffle == True:
		    np.random.shuffle(self.indexes)



	def __data_generation(self, batch_indexes):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
		mask = np.empty((self.batch_size, self.dim[0],self.dim[1]))
		y1 = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))

		# Generate data
		for ii in range(batch_indexes.shape[0]):
		    # Store sample
			file_id = batch_indexes[ii]//(self.nslices - self.crop[0] - self.crop[1])
			file_slice = batch_indexes[ii]%(self.nslices - self.crop[0] - self.crop[1])
			# Load data
			with h5py.File(self.list_IDs[file_id], 'r') as f:
				kspace = f['kspace']
                # Most volumes have 170 slices, but some have more. For these cases we crop back to 170 during training.
                # Could be made more generic.
				if kspace.shape[2] == self.dim[1]:
					X[ii,:,:,:] = kspace[self.crop[0]+file_slice]
				else:
					idx = int((kspace.shape[2] - self.dim[1])/2)
					X[ii,:,:,:] = kspace[self.crop[0]+file_slice,:,idx:-idx,:]
		X[:,:,145:,:] = 0 # Explicit zero-filling
		aux = np.fft.ifft2(X[:,:,:,::2]+1j*X[:,:,:,1::2],axes = (1,2))
		y1[:,:,:,::2] = aux.real
		y1[:,:,:,1::2] = aux.imag
		if self.shuffle:
		    idxs = np.random.choice(np.arange(self.under_masks.shape[0], dtype=int), self.batch_size, replace = True)
		else:
		    idxs = np.arange(0,self.batch_size,dtype = int)
		mask = self.under_masks[idxs]
		X[~mask,:] = 0
		aux2 = np.fft.ifft2(X[:,:,:,::2]+1j*X[:,:,:,1::2],axes = (1,2))
		X[:,:,:,::2] = aux2.real
		X[:,:,:,1::2] = aux2.imag
		norm = np.abs(aux2).max(axis = (1,2,3),keepdims = True) # Normalize using the maximum absolute value across channels.
                                                                # Could to be improved
      
        
		y1 = y1/norm  # Normalized fully sampled multi-channel reference. Could be converted to root sum of squares.
                      # it depends on how teams model the problem
		X = X/norm # Input is the zero-filled reconstruction. Suitable for image-domain methods. Change the code to not 
                   # compute the iFFT if input needs to be in k-space.
		return X, y1

