# -*- coding: utf-8 -*-
"""
Links:
    
https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_wavelet
https://nirpyresearch.com/wavelet-denoising-spectra/

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
 
# Generate synthetic signal and add WGN
pnts = 1000
x = np.sin(2*np.pi*1.4*np.arange(pnts)/pnts + 0.5) + np.sin(2*np.pi*3.5*np.arange(pnts)/pnts)

noise = np.random.normal(0,0.5,size=pnts)
xn = x + noise

# NOTES:
# Sigma set to None (default) -> calculates noise thresh automatically
# Method set to None (defaults to BayesShrink -> unique threshold for each subband)
x_den = denoise_wavelet(xn, wavelet='db4', mode='soft', wavelet_levels=6)
 
with plt.style.context(('seaborn-v0_8-whitegrid')):    
    plt.plot(xn,label='Noisy')    
    plt.plot(x_den, label='Denoised')    
    plt.plot(x,label='Orig')
    
    plt.xlabel("Time")
    plt.legend(loc='lower left')
plt.show()