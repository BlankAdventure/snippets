# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:04:31 2025

@author: BlankAdventure

A review of AWGN noise calculations in the context of power spectral density;
spectrograms; and complex signals. Includes discussion of the Log-Rayleigh
distribution which can be useful for estimating the NPSD.
"""

from scipy.signal import spectrogram
from scipy.fft import fftshift
import numpy as np
import matplotlib.pyplot as plt


fs = 20000 #Sampling rate [Hz]
dur = 1 #Duration in time [s]

# Next we specify the standard deviation of our Gaussian noise source.
# Importantly, note that sd^2 = variance = noise power [W]
sd = 0.001 #sd^2 = variance = noise power [w]


print (f'(1) Noise Power: {sd**2:.2e} [W] or {10*np.log10(sd**2):.1f} [dBW]')

pnts = fs*dur
nsr = np.random.normal(loc=0.0, scale=sd, size=pnts)
nsi = np.random.normal(loc=0.0, scale=sd, size=pnts)
nsz = nsr + 1j*nsi

# For completeness, lets check the noise power in our signals, using an 
# alternative definition. Notice that for the complex noise signal, the power 
# is doubled (and therefore higher by 3dB):
    
print (f'(2) Noise Power, Real: {np.mean(abs(nsr)**2):.2e} [W] or {10*np.log10(np.mean(abs(nsr)**2)):.1f} [dBW]')
print (f'(3) Noise Power, Imag: {np.mean(abs(nsi)**2):.2e} [W] or {10*np.log10(np.mean(abs(nsi)**2)):.1f} [dBW]')    
print (f'(4) Noise Power, Cmplx: {np.mean(abs(nsz)**2):.2e} [W] or {10*np.log10(np.mean(abs(nsz)**2)):.1f} [dBW]')


# Next we'll generate a spectrogram
nperseg = 512
noverlap = 384
win = 'boxcar'

f, t, sxx = spectrogram(nsz, fs, win, nperseg, noverlap, detrend=None, return_onesided=False,
                        scaling='density', mode='psd')
f = fftshift(f)

# Collapase the spectrogram along the time axis and plot:
psd_w = np.mean(sxx,axis=1)
No_est_db = 10*np.log10(np.mean(psd_w)) #Estimate from the plot

plt.figure()
plt.plot(f,10*np.log10(psd_w))
plt.axhline(y=No_est_db, color='r', linestyle='--', label='No_est')
plt.xlabel('Freq [Hz]')
plt.ylabel('PSD [dBW/Hz]')
plt.title('Power Spectral Density')
plt.grid()

print (f'(5) Estimated No from spectrogram: {No_est_db:.1f} [dBW/Hz]')

# Calculate the theoretical noise PSD, No. This formula should be fairly 
# intuitive: sd**2 is the noise power; we double it as we are combining two
# channels; and finally we take the bandwidth as fs, as we are computing
# the two-sided spectrogram, spanning -fs/2 to fs/2.
No_w = 2*sd**2/fs

print (f'(6) Theoretical NPSD: {10*np.log10(No_w):.1f} [dBW/Hz]')

# For completeness, here is the calculation to recover the noise power:
10**(No_est_db/10)*fs

#%% Next lets examine the lograyl distribution

from sklearn.neighbors import KernelDensity

def lograyl(u,x):    
    return (1/10)*np.log(10)*(10**(x/10))/(2*u**2)*np.exp(-(10**(x/10))/(2*u**2))       


x0 = 10*np.log10( abs(nsz)**2 ) #Power spectrum transformation


# We'll use KDE here, but could use a histogram instead
x_eval = np.linspace(-85, -40, num=50) 
kde = KernelDensity(kernel='exponential', bandwidth=0.4).fit(x0[:, np.newaxis])
ykde = kde.score_samples(x_eval[:, np.newaxis])

# Get the LR curve. Notice that we generate it with our specified sd value
ylr = lograyl(sd,x_eval)
peak = x_eval[np.argmax(ykde)]

plt.figure()
plt.plot(x_eval,np.exp(ykde),label='KDE')
plt.plot(x_eval,ylr,label='LR')
plt.xlabel('Noise Power [dBW]')
plt.axvline(x=peak, color='r', linestyle='--', label='Signal Power')
plt.legend()
plt.grid()

print (f'(7) KDE Peak Location: {peak:.1f} [dBW]')
print (f'(8) Expected Peak Location: {10*np.log10(2*sd**2):.1f} [dBW]')

# Notice that in the above we do not perform any actual fitting to the LR 
# distribution. Instead, the LR curve is generated directly from the same sd 
# used to generate the data, revealing the fact that the peak lands on 
# 10*np.log10(2*sd**2), which is the noise power [dBW]. Next, we show that the
# KDE curve naturally approximates the LR curve, and we can directly read the
# peak off it.