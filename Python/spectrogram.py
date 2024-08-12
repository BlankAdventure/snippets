# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 18:23:16 2024
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html#scipy.signal.ShortTimeFFT
https://www.stevejpurves.com/geoscience
"""
import numpy as np
from scipy.signal import spectrogram
from scipy.fft import fftshift
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# Wrapper function to consume the stft_generator and return the complete
# set of columns.
def build(func):
    def wrapper(*args, **kwargs):
        data = []
        times = []
        gen = func(*args, **kwargs)
        for g in gen:
            data.append(g[0])
            times.append(g[1])
        return np.array(data).transpose(), times
    return wrapper

# Homebrew implemenation 
@build
def stft_generator(sig, fs, chunk, nfft, overlap):
    win = np.hanning(chunk)
    num_samps = len(sig)
    new_samps = int(chunk*(1-overlap))
    rng = np.arange(0, chunk)
    tt = chunk / fs
    dt = new_samps / fs
    
    while (rng[-1] < num_samps):
        samps = sig[rng]
        col = np.fft.fftshift( np.fft.fft(samps*win, n=nfft) )
        col = np.flip(col,axis=0)
        yield col, tt
        rng = rng + new_samps
        tt = tt + dt


# Applies a CONSTANT phase shift across frequncies
# In principle, could be a function of freq
def fftshifter(x, phase_shift_in_radians):
    # Create an array to apply the coefficient to 
    # positive and negative frequencies appropriately
    N = len(x);
    R0 = np.exp(-1j*phase_shift_in_radians)
    R = np.ones(x.shape, dtype=complex)
    R[0]=0.
    R[1:N//2] = R0
    R[N//2:] = np.conj(R0)

    # Apply the phase shift in the frequency domain
    Xshifted = R*fft(x) # Could do this in time domain via convolution
    
    # Recover the shifted time domain signal
    y = np.real(ifft(Xshifted))
    
    return y



fs = 10000
dur = 1
t = np.arange(0,dur,1/fs)

f1 = 560
f2 = 4100

# Create a test signal
x = 1*np.sin(2*np.pi*f1*t + 0.5) + 0.5*np.sin(2*np.pi*f2*t)
noise = np.random.normal(0,1,size=len(x))
xn = x + noise

# Generate a complex signal -> set phase shift to 90 for perfect IMRR (analytic signal)
xs = fftshifter(xn, 75*np.pi/180 )
z = x + 1j*xs

# One way...
[ff,tt,sxx] = spectrogram(z,fs=fs,window='hann',nperseg=512,noverlap=384,return_onesided=False)
psd = np.mean(sxx,axis=1)

# Or this way, if we want the complex stft
#[stft, tt] = stft_generator(z, fs, 512, 512, 0.75)
#sxx = abs(stft)**2
#ff = np.linspace(fs/2,-fs/2,512) #This is not quite the right way to do this
#psd = np.mean( sxx, axis=1  )


fig = plt.figure(1, figsize=(5.5, 12))
plt.subplot(2, 1, 1)

plt.pcolormesh(tt, fftshift(ff), fftshift(sxx, axes=0), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

plt.subplot(2, 1, 2)
plt.plot(fftshift(ff),10*np.log10(fftshift(psd)))
plt.grid()