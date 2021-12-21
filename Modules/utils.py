import numpy as np


def sum_of_squares(img_channels: np.ndarray) -> np.ndarray:
    """Combines complex channels with square root sum of squares.

    :param img_channels: Complex channels
    :return: Combined image
    """
    sos = np.sqrt((np.abs(img_channels) ** 2).sum(axis=-1))
    return sos
