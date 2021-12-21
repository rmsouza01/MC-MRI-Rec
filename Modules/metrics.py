import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sewar.full_ref import vifp
from typing import Tuple


def metrics(rec: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray]:
    """Get SSIM, pSNR, and VIF metrics for a reconstruction / reference pair of volumes.

    :param rec: Numpy array containing an under-sampled reconstruction volume to evaluate.
    :type rec: np.ndarray
    :param ref: Numpy array containing the corresponding fully-referenced reconstruction volume.
    :type ref: np.ndarray
    :return: Tuple of np.ndarrays containing SSIM, pSNR, and VIF per slice.
        Length of each array is equal to number of slices in input volume
    :rtype: Tuple[np.ndarray]
    """
    ssim = np.zeros(rec.shape[0])
    psnr = np.zeros(rec.shape[0])
    vif = np.zeros(rec.shape[0])

    for ii in range(rec.shape[0]):
        data_range = np.maximum(ref[ii].max(), rec[ii].max()) - np.minimum(ref[ii].min(), rec[ii].min())
        ssim[ii] = structural_similarity(ref[ii], rec[ii], data_range=data_range)
        psnr[ii] = peak_signal_noise_ratio(ref[ii], rec[ii], data_range=data_range)
        vif[ii] = vifp(ref[ii], rec[ii], sigma_nsq=0.4)

    return ssim, psnr, vif
