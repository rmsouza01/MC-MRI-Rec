import numpy as np


def channel_wise_ifft(zero_filled_kspace: np.ndarray) -> np.ndarray:
    """Computes the iFFT across channels of multi-channel k-space data.

    The input is expected to be a complex numpy array.

    :param zero_filled_kspace: The zero-filled k-space
    :return: Channel wise iFFT
    """
    return np.fft.ifft2(zero_filled_kspace, axes=(1, 2))


def sum_of_squares(img_channels: np.ndarray) -> np.ndarray:
    """Combines complex channels with square root sum of squares.

    The channels are the last dimension (i.e., -1) of the input array.

    :param img_channels: Complex channels of input
    :return: Combined image sum of squares.
    """
    return np.sqrt((np.abs(img_channels) ** 2).sum(axis=-1))


def zero_filled_reconstruction(zero_filled_kspace: np.ndarray) -> np.ndarray:
    """Zero-filled reconstruction of multi-channel MR images.

    The input is the zero-filled k-space. The channels are the last dimension of the array. The input may be either
    complex-valued or alternate between real and imaginary channels in the last array dimension.

    :param zero_filled_kspace: Multi-channel input data containing either complex-valued data or complex data projected
    into the (real set)^2.
    :return: Zero filled reconstruction.
    """
    if not np.iscomplexobj(zero_filled_kspace):
        zero_filled_kspace = (
            zero_filled_kspace[:, :, :, ::2] + 1j * zero_filled_kspace[:, :, :, 1::2]
        )  # convert real-imag to complex data

    return sum_of_squares(channel_wise_ifft(zero_filled_kspace))
