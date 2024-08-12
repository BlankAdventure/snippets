# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:42:42 2024

"""
import numpy as np
#from sktime.datatypes import convert
import pandas as pd
import matplotlib.pyplot as plt

#SEED = 1111
def_rnd = np.random.default_rng()

def_opts = [(1,1.5,0.3),(0.5,4.7,1.2)]


clrs = ['r','b','g','k','y','c','m']


# Converts flat numpy array (examples, timepoints) into tsfresh df format
def flat_df (numpyflat: np.ndarray):
    dfs = []
    
    num_samples = numpyflat.shape[0]
    time_steps = np.arange(0, numpyflat.shape[1])

    # Construct DataFrames for each sample
    for q in range(num_samples):
        trace = numpyflat[q, :]        
        temp_df = pd.DataFrame({
            'time': time_steps,
            'y': trace,
            'id': q+1
        })
        
        # Append to the list
        dfs.append(temp_df)

    # Concatenate all DataFrames at once
    df = pd.concat(dfs, ignore_index=True)
    return df

# Converts flat (i.e., 2D) numpy array into the sktime df format
# def np_to_sk (numpyflat):
#     n3d = convert(numpyflat, from_type="numpyflat", to_type="numpy3D")
#     sk = convert(n3d, from_type="numpy3D", to_type="pd-multiindex")
#     return sk


# Generates clusters of traces
# 'flat': returend trace arrays are 2D arrays of size (n_clusters x n_samps, pnts)
# 'stacked': returend trace arrays are 3D arrays of size (n_clusters, n_samps, pnts)
def trace_clusters(n_samps, n_clusters=2, n_terms=2, ns_lvl=0.2, pnts=128, mode=None, plot=False, sf=0.1, seed=None):
    global def_rnd
    if seed is not None:
        def_rnd = np.random.default_rng(seed)
    
    clean_out = []
    noisy_out = []
    targs = []
    for idx in range(0, n_clusters):    
        A = def_rnd.uniform(0,1,n_terms)
        F = def_rnd.uniform(0,5,n_terms)
        PH = def_rnd.uniform(0,2*np.pi,n_terms)
        G = np.vstack((A,F,PH))
        c = [tuple(G[:,x]) for x in range(0,n_terms)]    
        clean = get_samples(n_samps, c, pnts, sf)
        noisy = add_noise(clean, ns_lvl)        
        clean_out.append(clean)
        noisy_out.append(noisy)           
        
        
    clean_out = np.array(clean_out)
    noisy_out = np.array(noisy_out)
    targs = np.array(range(0, n_clusters))
    
    if plot:
        with plt.style.context(('seaborn-v0_8-whitegrid')):
            for x in range(noisy_out.shape[0]):
                plt.plot(noisy_out[x,:].transpose(), clrs[x])
        plt.show()

    if mode == 'flat':
        clean_out = np.reshape(clean_out, (n_samps*n_clusters, pnts))
        noisy_out = np.reshape(noisy_out, (n_samps*n_clusters, pnts))
        targs = np.array([[x]*n_samps for x in targs]).flatten()

    return clean_out, noisy_out, targs


def add_noise(clean_traces: list, ns_lvl: float):
    """Add ns_lvel WGN to clean traces"""
    
    out = []    
    for trace in clean_traces:
        noisy = trace + def_rnd.normal(0,ns_lvl,size=len(trace))
        out.append(noisy)
    return out

def get_signal(n_samples: int, fs:float=100, dur:float=1, sources:list=def_opts, sf:float=0.1):
    dt = 1/fs
    t = np.arange(0,dur,dt)
    traces = []       
    for _ in range(0, n_samples):
        x = 0
        for (a,f,ph) in sources:
            a_r = def_rnd.uniform(a-(a*sf), a+(a*sf))
            f_r = def_rnd.uniform(f-(f*sf), f+(f*sf))
            ph_r = def_rnd.uniform(ph-(ph*sf), ph+(ph*sf))
            x = x + a_r*np.sin(2*np.pi*f_r*t + ph_r) 
        traces.append(x)
    return (t,traces)
    
    
    
def get_samples(n_samples: int, sources:list=def_opts, pnts: int=128, sf: float=0.1):
    """Generate random data traces"""

    t = np.linspace(0, 1, pnts)
    traces = []       
    for _ in range(0, n_samples):
        x = 0
        for (a,f,ph) in sources:
            f_r = def_rnd.uniform(f-(f*sf), f+(f*sf))
            ph_r = def_rnd.uniform(ph-(ph*sf), ph+(ph*sf))
            a_r = def_rnd.uniform(a-(a*sf), a+(a*sf))
            x = x + a_r*np.sin(2*np.pi*f_r*t + ph_r) 
        traces.append(x)
    return traces
