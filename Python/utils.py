# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:51:47 2024

@author: BlankAdventure
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.fft import fftshift, ifftshift
from scipy.signal import spectrogram, hilbert, windows, medfilt, resample
from scipy import interpolate
from scipy.fftpack import fft, ifft, fftfreq

from nptyping import NDArray, Shape
from typing import Any, List

vector =  NDArray[Shape["*"], Any]

class Channel():
    '''
    Channel models the response of a single channel/filter. It is defined
    by a specified fs, mag_resp and ph_resp. Signals can be filtered in either
    the time domain or frequency domain.    
    '''
    
    def __init__(self, fs: int, mag_resp: dict, ph_resp: dict) -> None:
        '''
        Initializes a new Channel object. 
        '''
        
        self.fs = fs
        self.mag_resp = mag_resp
        self.ph_resp = ph_resp
        self.freqs:dict = {}
    
    def build_resp(self, N: int, do_plots: bool=False) -> None:
        '''
        Builds an N-point frequency response by interpolating the mag_resp
        and ph_resp points provided during initialization. Optionally plots
        the frequency response.
        '''
        
        freqs = build_freq_resp(self.fs, N, self.mag_resp, self.ph_resp, do_plots, force_real=True)
        self.freqs = freqs 
        
    def apply_fd(self, sig: vector) -> vector:
        '''
        Filters the input signal sig through the channel's frequency response,
        in the frequency domain. This is equivlalent to *circular* convolution
        in time.
        '''
        
        if self.freqs:
            if len(sig) != len(self.freqs['z_full']):
                print('Supplied signal length does not match existing FreqResp. Building new.')
                self.build_resp(len(sig))
        else:
            print ('No existing FreqResp. Building new.')
            self.build_resp(len(sig))    
        
        sig_out = ifft(self.freqs['z_full']*fft(sig))
        
        return np.real(sig_out)
        
    def apply_td(self, sig: vector, N: int=0, nk: int=0, do_shift: bool=True) -> vector:     
        '''
        Filters the input signal sig through the channel's frequency response
        in the time domain. That is, it performs a linear convolution in time.
        '''

        match [bool(self.freqs), N > 0]:
            case [True, True]:
                print('Building new FreqResp using N')
                self.build_resp(N)
            case [True, False]:
                print('Using existing FreqResp')
            case [False, True]:
                print('Building new FreqResp using N')
                self.build_resp(N)
            case [False, False]:
                print('Building new FreqResp using len(sig)')
            
        FIR = np.real(ifft(self.freqs['z_full']))
        fir_trunc = get_truncated(FIR, nk)
        self.FIR = FIR
        self.fir_trunc = fir_trunc
        sig_out = apply_fir(self.fir_trunc, sig, do_shift)
        return sig_out
        
    def plot_resp(self, N: int=0) -> None:
        '''
        Plot the channel's N-point frequency response. If N is not specified it uses
        the existing response.
        '''
        
        if N > 0:
            build_freq_resp(self.fs, N, self.mag_resp, self.ph_resp, do_plots=True)
        else:
            if self.freqs:
                N = len(self.freqs['fi'])
                build_freq_resp(self.fs, N, self.mag_resp, self.ph_resp, do_plots=True)
            else:
                print('No FreqResp to show. Create one, or specify N.')



class Quadrature():
    '''
    Quadrature models an imperfect quadrature system, and demonstrates a mechanism
    for fixing this imperfection.
    
    fs -> sampling rate [Hz]
    chan_dict -> channel dictionary containing desired mag & phase responses
    for each of the I and Q channels. 
    '''
    
    def __init__(self, fs: int, chan_dict: dict) -> None:
        '''
        Creates a new Quadrature object. 
        
        fs -> sampling rate
        chan_dict -> dictionary containing desired I & Q channel response to 
        be corrected.
        '''
        
        self.fs = fs
        self.chan_dict = chan_dict
        self.FIR:None|vector = None
        self.Zcorr:vector #None|vector = None
        self.chanI = Channel(fs, *chan_dict['I'])     
        self.chanQ = Channel(fs, *chan_dict['Q']) 
        
    def inject_signal(self, USB: vector, LSB: vector|None|float) -> vector:
        '''
        Injects an USB and LSB signal into the model and returns the I  & Q
        signal vectors. The function first creates a 'perfect' IQ quadrature
        signal via the Hilbert transform then passes these signals through the
        specified channel models to cause distortion.
        '''
        
        match LSB:
            case float():
                print('Filling channel with noise')
                LSB = np.random.randn(len(USB))*LSB
            case None:
                print('Noise-free channel')
                LSB = np.zeros(len(USB))
        I, Q = get_iq(USB, LSB)        
        Ip = self.chanI.apply_fd(I)
        Qp = self.chanQ.apply_fd(Q)        
        return Ip + 1j*Qp

    def sys_ident(self, USB: vector, LSB: vector, plot: bool=False, ftest: List[float]=[], chunk: int|None=None, noverlap: int|None=None, window_name: str='hann', med_filt: int=0) -> None:
        '''
        Determines the frequency correction needed to match Q to I, and impose
        the expected 90 degree phase shift. The ideal input should be a 
        broadband signal in USB and zero in LSB. 
        
        If a value is provided for chunk, then it uses a sliding window approach
        for determing the required correction filter. This can be useful when
        noise is present in the probe signal (which for real applications it 
                                              would be!)
        '''
        
        Zp = self.inject_signal(USB, LSB)
        Ip = np.real(Zp)
        Qp = np.imag(Zp)
        
        # Peform system identification. This yields hxy, which represents the 
        # FIR filter needed to match Q to I...
        if chunk is not None:
           hxy, Hxy = sys_ident_window(Ip,Qp,chunk,noverlap,window_name,med_filt)
        else:
           hxy, Hxy = sys_ident(Ip,Qp) 
        
        self.hxy = hxy
        self.Hxy = Hxy
        
        N = len(self.Hxy)
        
        # ...however We don't to just match Q to I. We want to impose a 90-degree
        # phase shift between the two, so we apply some furter modfication to 
        # the resonse below.
        
        Mk = 1/abs(self.Hxy)            
        ph_p_corr = np.angle(self.Hxy[0:N//2]) + np.pi/2
        ph_n_corr = np.angle(self.Hxy[N//2:]) - np.pi/2            
        Phk = np.concatenate((ph_p_corr, ph_n_corr))
        self.Zcorr = Mk*np.exp(-1j*Phk)
        
        # Enforce proper symmetry
        self.Zcorr[0] = 0 #np.real(self.Zcorr[0]) 
        self.Zcorr[N//2] = 0 #np.real(self.Zcorr[N//2])         
        self.FIR = np.real(ifft(self.Zcorr))
        
        if plot: 
            self.plot_hxy(ftest=ftest)   
            plot_resp(self.fs, self.FIR)
    
    def correct_fd(self, z_in: vector) -> None|vector:
        '''
        Apply quadrature correction in the frequency domain. z_in should be
        complex signal returned by inject_signal.
        '''

        if self.Zcorr is not None:            
            N_sig = len(z_in)
            N_fir = len(self.Zcorr)
            
            ip = np.real(z_in)
            qp = np.imag(z_in)            

            if N_sig > N_fir:                
                freq_dict = freq_points(self.fs, N_fir, self.Zcorr)  
                mag_resp = {'x': freq_dict['f_pos'], 'y': np.abs(freq_dict['z_pos'])}
                ph_resp  = {'x': freq_dict['f_pos'], 'y': -np.rad2deg(np.angle(freq_dict['z_pos']))}
                ZR = build_freq_resp(self.fs, N_sig, mag_resp, ph_resp)['z_full']    
            else:
                ZR = self.Zcorr            
            out = ip + 1j*np.real(ifft(fft(qp)*ZR))
            return out
        return None
        
 
    
    def correct_td(self, z_in: vector, nk: int=0) -> None|vector:
        '''
        Apply quadrature correction in the time domain. z_in should be complex 
        signal returned by inject_signal.
        '''
        
        if self.FIR is not None:
            return apply_fir_quadrature(self.FIR, z_in, nk=nk)
        else:
            return None
    
    def plot_hxy (self, ftest: list[float]=[]) -> None:
        '''
        Plot the determined Q/I ratio.
        '''
        
        plot_hxy(self.fs, self.hxy, ftest=ftest)
        
    
                
      
def symm_check(K:vector, rtol:float=0, atol:float=1e-5) -> bool:
    '''
    Checks symmetry properties of K to ensure its inverse transform
    yields a purely real result.
    '''
    N = len(K)
    
    R1 = K[1:(N//2)]
    R2 = K[(N//2)+1:]
    
    DC = K[0]
    IP = K[N//2]
    
    is_symmetric = True
    
    if not np.isclose( np.imag(DC),0, rtol, atol):
        print('WARNING: N=0 is NOT real!')
        is_symmetric = False
        
    if not np.isclose( np.imag(IP),0, rtol, atol):
        print('WARNING: N/2 is NOT real!')
        is_symmetric = False
    
    cvect = np.isclose(R1,np.flipud(R2.conjugate()),rtol, atol)
    if not all(cvect):
        print('WARNING: Imperfect conjugate symmetry!')
        is_symmetric = False
    
    return is_symmetric

def get_truncated(fir:vector, nk:int=0) -> vector:
    '''
    Returns a truncated version of the supplied FIR filter.    
    nk -> number of taps to retain.
    '''
    
    if nk <= 0:
        nk = len(fir)
    fir_trunc = np.concatenate( (fir[0:nk//2], fir[-nk//2:]))
    return fir_trunc
    

def apply_fir(fir:vector, sig:vector, do_shift:bool) -> vector:
    '''
    Filters a signal in time with an FIR filter - that is the convolution of 
    the signal sequence and FIR taps. Note that this yields the linear convolution
    whereas multiplying in the frequency domain yields the circular convolution.
    
    fir -> vector containing the FIR coefficients
    sig -> signal vector
    do_shift -> Boolean flag to shift the position of the taps by N//2+1.
    '''
    
    if do_shift:
        fir = np.roll(fftshift(fir),[0,-1])
    
    if len(sig) > len(fir):
        sig_out = np.convolve(sig,fir,mode='same')
    else:
        sig_out = np.convolve(fir,sig,mode='same')
    return sig_out
        
def apply_fir_quadrature (fir:vector, z:vector, nk:int=0) -> vector:
    fir_t = get_truncated(fir, nk)
    
    delay = np.zeros(len(fir_t))
    delay[0] = 1    
    
    iif = apply_fir(delay, np.real(z), do_shift=True)
    qqf = apply_fir(fir_t, np.imag(z), do_shift=True)
    
    return iif + 1j*qqf    
    


def get_iq(USB:vector, LSB:vector) -> tuple[vector, vector]:
    '''
    Generates idealized I and Q signals from specified LSB and USB signals.
    '''
    
    I = np.imag(hilbert(LSB)) + USB;
    Q = LSB + np.imag(hilbert(USB));
    return I,Q

def freq_points(fs:float, N:int, z:vector|List[float]=[]) -> dict:
    '''
    Helper function that returns a dictionary consisting of the FFT frequencies
    split into their positive and negative ranges, along with an input signal
    if supplied.
    '''
    
    dt = 1/fs
    fi = fftfreq(N,dt)
    f_pos = fi[0:N//2]
    f_neg = fi[N//2:]    
    z_pos = z[0:N//2]
    z_neg = z[N//2:]        
    return {'fi': fi, 'f_pos': f_pos, 'f_neg': f_neg, 'z_pos': z_pos, 'z_neg': z_neg}


def build_freq_resp (fs:float, N:int, mag_resp, ph_resp, do_plots:bool=False, window_name:str='boxcar', force_real:bool=False) -> dict:
    ''' 
    Generates an arbitrary complex frequency response via interpolation.
    Response is hermitian such that the impulse response is real. Returns the
    interpolated frequency response in a dictionary.
    
    fs -> sampling frequency [Hz]
    N -> number of points to interpolate to.
    mag_resp -> dictionary of magnitude response (x,y points)
    ph_resp -> dictionary of phase response (x,y poits)
    do_plots -> True/false to dispaly the spectrum.
                              
    '''         
    window = windows.get_window(window_name, N//2)
    
    freqs = freq_points(fs,N)
    f_pos = freqs['f_pos']
    f_neg = freqs['f_neg']
    
    # Interpolate over the positive frequencies
    mi_p =  interpolate.PchipInterpolator(mag_resp['x'], mag_resp['y'])(f_pos)
    phi_p = interpolate.PchipInterpolator( ph_resp['x'],  ph_resp['y'])(f_pos)
    zp = mi_p*np.exp(-1j*np.deg2rad(phi_p))*window

    # Interpolate over the negative frequenies
    mi_n =  interpolate.PchipInterpolator(mag_resp['x'], mag_resp['y'])(np.negative(f_neg))
    phi_n = interpolate.PchipInterpolator( ph_resp['x'],  ph_resp['y'])(np.negative(f_neg))
    zn = np.conj(mi_n*np.exp(-1j*np.deg2rad(phi_n)))*window

    # Combine to make double-sided spectrum
    z_full = np.concatenate((zp,zn))
    
    if force_real:
        z_full[0] = np.real(z_full[0])
        z_full[N//2] = np.real(z_full[N//2])
    
    freqs['z_full'] =  z_full   
    
    if do_plots:
        IR = np.real(ifft(z_full))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4) )
        ax1.plot(f_pos, mi_p)
        ax1.scatter(mag_resp['x'], mag_resp['y'])
        ax1.set_xlabel('Freq [Hz]'); ax1.set_ylabel('Magnitude')
        ax1.grid()
        ax1.set_xlim([0, fs/2])
        
        ax2.plot(f_pos, phi_p)
        ax2.scatter(ph_resp['x'], ph_resp['y'])
        ax2.grid()
        ax2.set_xlabel('Freq [Hz]'); ax2.set_ylabel('Phase (Deg)')    
        ax2.set_xlim([0, fs/2])

        ax3.plot(np.real(IR))
        ax3.grid()
        ax3.set_xlabel('Taps'); ax3.set_ylabel('Value')  
        plt.tight_layout()
    
    return freqs



def plot_resp(fs: float, fir:vector, ftest:List[float]=[]) -> None:
    '''
    Plot magnitude and phase response of FIR filter.
    '''
    
    Z = fft(fir)   
    freqs = freq_points(fs, len(Z), Z)    
    f_pos = freqs['f_pos']
    mag = np.abs(freqs['z_pos'])
    ph = np.rad2deg(np.angle(freqs['z_pos']))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4) )
    ax1.plot(f_pos, mag)
    ax1.set_xlabel('Freq [Hz]') 
    ax1.set_ylabel('A.U.')
    ax1.set_title('Magnitude CorrFact')
    ax1.grid()
    ax1.set_xlim([0, fs/2])
    ax1.set_ylim([0.5, 1.5])
    
    ax2.plot(f_pos, ph)
    ax2.grid()
    ax2.set_xlabel('Freq [Hz]');
    ax2.set_ylabel('Degrees')
    ax2.set_title('Phase CorrFact')    
    ax2.set_xlim([0, fs/2])
    ax2.set_ylim([-15, 15])

    ax3.plot(fir)
    ax3.grid()
    ax3.set_title('IR')
    ax3.set_xlabel('Taps'); ax3.set_ylabel('Value')  
    plt.tight_layout()
    
def plot_hxy(fs:float, hxy:vector, ftest:List[float]=[]) -> None:    
    ''' 
    Plots system impulse response defined in hxy.    
    ftest -> list of frequency markers to draw on plot (optional)
    '''
    
    N = len(hxy)
    dt = 1/fs
    fi = fftfreq(N,dt)
    f_pos = fi[0:N//2]
    
    Hxy = fft(hxy)
    mag = np.abs(Hxy[0:N//2])
    ph = np.rad2deg(np.angle(Hxy[0:N//2]))
    imrr = IMRR(mag, 90+ph)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4), sharex=True)
    
    ax1.set_title('Magnitude Ratio')
    ax1.set_xlabel('F (Hz)')
    ax1.set_ylabel('A.U.')
    ax1.set_xlim([0, (fs/2)])
    ax1.set_ylim([0.5, 1.5])
    ax1.axhline(1,color='k')
    [ax1.axvline(f,color='r') for f in ftest]
    ax1.plot(f_pos,mag); ax1.grid()    
    
    ax2.set_title('Phase Difference')
    ax2.set_xlabel('F (Hz)')
    ax2.set_ylabel('Degrees') 
    ax2.set_xlim([0, (fs/2)])
    ax2.set_ylim([-110, -70])
    ax2.axhline(-90,color='k'); 
    [ax2.axvline(f,color='r') for f in ftest]
    ax2.plot(f_pos,ph); ax2.grid()
    
    ax3.set_title('IMRR')
    ax3.set_xlabel('F (Hz)')
    ax3.set_ylabel('dB')  
    ax3.set_xlim([0, (fs/2)])
    [ax3.axvline(f,color='r') for f in ftest]
    ax3.plot(f_pos,-imrr); ax3.grid()
    plt.tight_layout()    
        
    
def pass_thru(fs:float) -> tuple[dict,dict]:
    '''
    Generates a 'pass through' channel that leaves the signal unmodified.
    '''
    
    mag_resp = {'x': [0, fs/2], 'y': [1, 1] }
    ph_resp =  {'x': [0, fs/2], 'y': [0, 0] }
    return mag_resp, ph_resp

def imperfect_hilbert(fs:float) -> tuple[dict,dict]:
    '''
    Generates an imperfect hilbert filter.
    '''
    
    f1 = 1*(fs/2)/3
    f2 = 2*(fs/2)/3
    f3 = 3*(fs/2)/3
    mag_resp = {'x': [0, f1, f2, f3], 'y': [1.5,1.03,0.97,0.6] }
    ph_resp =  {'x': [0, f1, f2, f3], 'y': [80,92,88,100] }
    return mag_resp, ph_resp

def imperfect_pass(fs:float) -> tuple[dict,dict]:
    '''
    Adds a fixed linear error to both the magnitude and phase.
    '''
    
    mag_resp = {'x': [0, fs/2], 'y': [1.2, 0.90] }
    ph_resp =  {'x': [0, fs/2], 'y': [5, -10] }
    return mag_resp, ph_resp

def add_error(fs:float) -> tuple[dict,dict]:
    '''
    Adds a fixed non-linear error to both the magnitude and phase.
    '''

    f1 = 1*(fs/2)/3
    f2 = 2*(fs/2)/3
    f3 = 3*(fs/2)/3
    mag_resp = {'x': [0, f1, f2, f3], 'y': [1.5,1.03,0.97,0.6] }
    ph_resp =  {'x': [0, f1, f2, f3], 'y': [10,-5,5,-10] }
    return mag_resp, ph_resp


def add_offset(fs:float) -> tuple[dict,dict]:
    '''
    Adds a fixed linear error to both the magnitude and phase.
    '''
    mag_resp = {'x': [0, fs/2], 'y': [1.10, 0.90] }
    ph_resp =  {'x': [0, fs/2], 'y': [-10, 10] }
    return mag_resp, ph_resp

def random_error(fs:float, pnts:int, max_var:float) -> tuple[dict,dict]:
    '''
    Returns a channel response with random magnitude & phase error.

    Parameters
    ----------
    fs : float
        Sampling rate [Hz]
    pnts : int
        # of random frequency points to use. Smaller value produces smoother
        responses.
    max_var : float
        max % variation from 0-value to allow.

    Returns
    -------
    Tuple[dict,dict]
        mag_resp and ph_resp dictionaries.

    '''
    mag_resp, ph_resp = random_channel(fs, pnts, max_var)
    ofs = ph_resp['y'] - 90
    ph_resp['y'] = ofs
    return mag_resp, ph_resp
    
def lowpass(fs:float, f1:float, f2:float) -> tuple[dict,dict]:   
    '''
    Basic minimum-phase lowpass filter.

    Parameters
    ----------
    fs : float
        Sampling rate [Hz]
    f1 : float
        Transition band start [Hz]
    f2 : float
        Transition band stop [Hz]

    Returns
    -------
    tuple[dict,dict]
        mag_resp and ph_resp dictionaries.

    '''
    mag_resp = {'x': [0, f1, f2, fs/2], 'y': [1, 1,0,0] }
    ph_resp =  {'x': [0, fs/2], 'y': [0, 0] }
    return mag_resp, ph_resp
    

def hilbert_resp (fs:float) -> tuple[dict,dict]:
    """
    Generates a channel frequency response consisting of the ideal hilbert
    transform.

    Parameters
    ----------
    fs : float
        sampling rate [Hz]

    Returns
    -------
    tuple[dict,dict]
        mag_resp and ph_resp dictionaries.

    """
    
    mag_resp = {'x': [0, fs/2], 'y': [1, 1] }
    ph_resp =  {'x': [0, fs/2], 'y': [90, 90] }
    return mag_resp, ph_resp

def simple_shift (fs:float, ds:int) -> tuple[dict,dict]:
    '''
    Generates a channel frequency response consisting of a simple time shift.

    Parameters
    ----------
    fs : float
        samping rate [Hz]
    ds : int
        # of samples to shift by

    Returns
    -------
    tuple[dict,dict]
        mag_resp and ph_resp dictionaries.

    '''
    mag_resp = {'x': [0, fs/2], 'y': [1, 1] }
    ph_resp =  {'x': [0, fs/2], 'y': [0, ds*360/2] }
    return mag_resp, ph_resp

def random_channel(fs:float, pnts:int, max_var:float) -> tuple[dict,dict]:
    '''
    Generates a random channel frequency response.
    
    fs -> sampling rate [Hs]
    pnts -> # of random frequency points to generate (better if kept low)
    max_var -> % maximum variation from ideal (1+/-max_var / 90d +/- max_var)
    
    Returns a tuple consisting of the magnitude points and phase points.
    '''
    
    f1 = np.random.random(pnts)*fs/2
    f1.sort()
    f1 = np.concatenate([[0],f1,[fs/2]])
    y1 = 1 + np.random.randn(pnts+2)*1*max_var
    mag_resp = {'x': f1, 'y': y1}
    
    f2 = np.random.random(pnts)*fs/2
    f2.sort()
    f2 = np.concatenate([[0],f2,[fs/2]])
    y2 = 90 + np.random.randn(pnts+2)*90*max_var
    ph_resp = {'x': f2, 'y': y2}    
    return mag_resp, ph_resp


def IMRR (mag_ratio:vector, ph_err:vector) -> vector:
    ''' 
    Calculate the Image Rejection Ratio
    '''    
    IMRR = (mag_ratio**2 + 1 + 2*mag_ratio*np.cos(np.deg2rad(ph_err))) / (mag_ratio**2 + 1 - 2*mag_ratio*np.cos(np.deg2rad(ph_err)))
    #IMRR[IMRR == 0] = np.finfo(float).eps
    return 10*np.log10(IMRR)


def quick_psd(sig:vector, fs:float, window:str='hann', nperseg:int=512, overlap:float=0.75, plot_mode:int=2, vrng:tuple[None|float,None|float]=(None,None)) -> None:
    ''' 
    Plot the PSD or spectrogram of supplied signal.
    
    plot_mode = 0 -> PSD & spectrogram (default)
    plot_mode = 1 -> PSD
    plot_mode = 2 -> Spectrogram
    '''    
    
    noverlap = int(nperseg*overlap)
    [ff,tt,sxx] = spectrogram(sig,fs=fs,window='hann',nperseg=nperseg,noverlap=noverlap,return_onesided=False)
    psd = np.mean(sxx,axis=1)
    
    match plot_mode:
        case 0:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
            ax1.plot((fftshift(ff)),10*np.log10(fftshift(psd)))
            ax1.set_ylabel('[dBW/Hz]')
            ax1.set_xlabel('Freq [Hz]')
            ax1.grid()            
            ax1.set_xlim([-fs/2, fs/2])
            
            ax2.pcolormesh(tt, fftshift(ff), fftshift(10*np.log10(sxx), axes=0), shading='gouraud', vmax=vrng[0], vmin=vrng[1])
            ax2.set_ylabel('Frequency [Hz]')
            ax2.set_xlabel('Time [sec]')    
            ax2.set_ylim([-fs/2, fs/2])
            plt.tight_layout()
            
        case 1:
            fig, ax = plt.subplots(1)
            ax.plot(fftshift(ff),10*np.log10(fftshift(psd)))
            ax.grid()
            ax.set_ylabel('[dBW/Hz]')
            ax.set_xlabel('Freq [Hz]')

        case 2:
            fig, ax1  = plt.subplots(1) # figsize=(6,6))
            ax1.pcolormesh(tt, fftshift(ff), fftshift(10*np.log10(sxx), axes=0), shading='gouraud',vmax=vrng[0], vmin=vrng[1])
            ax1.set_ylabel('Frequency [Hz]')
            ax1.set_xlabel('Time [sec]')
                

def dem_matrix (a:vector, n:int, noverlap:int) -> NDArray:
    ''' 
    Generates the delay-embeded matrix for input vector a.
    
    a -> input vector (1D)
    n -> chunk size
    noverlap -> # of overlapping samples in each chunk.
    
    Output: a p x n matrix where n is the chunk size and p depends on the input
    parameter values.
    '''
    
    N = len(a)
    ofs = n - noverlap
    k = int(np.fix((N - noverlap) / (n - noverlap)))

    # Create an array of indices for the windows
    indices = np.array([i * ofs for i in range(k)])[:, None] + np.arange(n)

    # Using the indices to create the output array
    x = a[indices]

    # Reshape x to have the correct shape
    x = x.reshape(k, n)
    return x

def sys_ident_window(sig_in:vector, sig_out:vector, chunk:int, noverlap:int|None=None, window_name:str='boxcar', med_filt:int=0) -> tuple[vector,vector]:
    '''
    Performs system identification in the frequency domain using a sliding
    window approach. Final result is an average across each analysis block.
    
    sig_in -> probe/input signal (should be broadband in freq domain)
    sig_out -> response vector
    chunk -> # samples to use in each analysis block
    noverlap -> # samples to overlap between each block
    window_name -> name of window to apply to each chunk
    
    Returns a tuple consisting of the determined impulse response h[t] and freq
    response H[w].
    '''
    
    if noverlap is None:
        noverlap = chunk - 1
    
    window = windows.get_window(window_name, chunk)
    
    X = dem_matrix(sig_in, chunk, noverlap)
    X = np.multiply(X, window)
    X = fft(X,axis=1)
    
    Y = dem_matrix(sig_out, chunk, noverlap)
    Y = np.multiply(Y, window)
    Y = fft(Y,axis=1)
    
    Rxx = np.conj(X)*X
    Rxy = np.conj(X)*Y
    
    Hxy = Rxy / Rxx   
    
    Hxy = np.mean(Hxy,axis=0)

    if med_filt > 0:
        Hxy = medfilt(np.real(Hxy), kernel_size=med_filt) + 1j*medfilt(np.imag(Hxy), kernel_size=med_filt)
    
    hxy = np.real(ifft(Hxy))

    return (hxy, Hxy)
    
def sys_ident(sig_in:vector, sig_out:vector) -> tuple[vector,vector]:
    '''
    Performs system identification in the frequency domain.
    
    sig_in -> probe/input signal (should be broadband in freq domain)
    sig_out -> response vector
    
    Returns a tuple consisting of the determined impulse response h[t] and freq
    response H[w].
    '''
    
    X = fft(sig_in)
    Y = fft(sig_out)

    Rxx = np.conj(X)*X
    Rxy = np.conj(X)*Y

    Hxy = Rxy / Rxx
    hxy = ifft(Hxy)
    
    return (hxy, Hxy)

def osc_fm(fs:float, dur:float, A:float, f_min:float, f_max:float, f_slow:float) -> vector:
    '''
    Returns a sinusoidally-varying FM test tone.
    
    fs -> sampling rate [Hz]
    dur -> duration [s]
    A -> amplitude [v]
    f_min -> minimum frequency [Hz]
    f_max -> maximum frequency [Hz]
    f_slow -> osciallation frequency [Hz]
    '''
    
    t = np.arange(0,dur,1/fs)
    fm = A*np.cos(2*np.pi*(((f_max-f_min)/2)*(t+np.cumsum(np.sin(2*np.pi*f_slow*t))/fs)+t*f_min))
    return fm

    

    
  

