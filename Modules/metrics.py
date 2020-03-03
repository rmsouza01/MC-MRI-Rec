import numpy as np
import h5py
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sewar.full_ref import vifp

def metrics(rec,ref):

    ssim = np.zeros(rec.shape[0])
    psnr = np.zeros(rec.shape[0])
    vif = np.zeros(rec.shape[0])
    
    for ii in range(rec.shape[0]):
        ssim[ii] = structural_similarity(ref[ii],rec[ii],data_range=(ref[ii].max()-ref[ii].min()))
        psnr[ii] = peak_signal_noise_ratio(ref[ii],rec[ii],data_range=(ref[ii].max()-ref[ii].min()))
        vif[ii] =  vifp(ref[ii],rec[ii],sigma_nsq = 0.4)

    return ssim,psnr,vif
