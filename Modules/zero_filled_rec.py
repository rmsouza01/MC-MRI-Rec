import numpy as np



def channel_wise_ifft(zero_filled_kspace):
    """
    Computes the iFFT across channels of multi-channel k-space data. The input is expected to be a complex numpy array.
    """
    return np.fft.ifft2(zero_filled_kspace, axes = (1,2))
    
    

def sum_of_squares(img_channels):
    """
    Combines complex channels with square root sum of squares. The channels are the last dimension (i.e., -1) of the input array.
    """
    return np.sqrt((np.abs(img_channels)**2).sum(axis = -1))
    return sos    

def zero_filled_reconstruction(zero_filled_kspace):
    """
    Zero-filled reconstruction of multi-channel MR images. The input is the zero-filled k-space. The channels
    are the last dimension of the array. The input may be either complex-valued or alternate between real and imaginary channels 
    in the last array dimension.
    """
    if np.iscomplexobj(zero_filled_kspace):
        zero_filled_kspace = zero_filled_kspace[:,:,:,::2] + 1j*zero_filled_kspace[:,:,:,1::2] #convert real-imag to complex data
    
    return sum_of_squares(channel_wise_ifft(zero_filled_kspace)) 
    