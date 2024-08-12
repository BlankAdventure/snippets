# -*- coding: utf-8 -*-
"""
Demonstrates basic IIR and FIR filtering strategies. 

https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

## ******* Second-Order Section IIR examples  *******  ##
# Can use other filter prototypes, cheby, etc.


def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

# Generate synthetic signal and add WGN
pnts = 1000
t = np.arange(pnts)/pnts
fs = 1/(t[1]-t[0])
x = np.sin(2*np.pi*1.4*t + 0.5) + np.sin(2*np.pi*3.5*t)
noise = np.random.normal(0,0.5,size=pnts)
xn = x + noise
 
#%% ***** SOS IIR example *****
# (1) Filter signal with IIR filer
filtered = lowpass(xn,10,fs)


#%% ***** Convolution, numpy-only *****

kernel_width_seconds = 0.1
kernel_size_points = int(kernel_width_seconds * fs)
kernel = np.hanning(kernel_size_points)

# normalize the kernel
kernel = kernel / kernel.sum()

#%% (2) Filter signal by convolving kernel with the signal
filtered = np.convolve(kernel, xn, mode='same')

#%% (3) Filter by appling filtfilt method
filtered = scipy.signal.filtfilt(kernel,1,x)


#%%
with plt.style.context(('seaborn-v0_8-whitegrid')):    
    plt.plot(t,xn,label='Noisy')    
    plt.plot(t,filtered, label='Denoised')    
    plt.plot(t,x,label='Orig')
    
    plt.xlabel("Time [s]")
    plt.ylabel('Amplitude')
    plt.legend(loc='lower left')
plt.show()