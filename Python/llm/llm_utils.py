# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:16:01 2025

@author: BlankAdventure
"""


import random
import numpy as np
import matplotlib.pyplot as plt

Paired = list[tuple[list[np.ndarray],int]]

# Helper function for stringifying a list of floats
seq_enc = lambda v: ','.join([f'{x:.2f}' for x in v]) 

def supervised_prompt(d: Paired, prompt: str = '') -> str:
    """
    Generate a supervised prompt of the form:
        
    '''
    SEQUENCE: <sequence>
    CLASS: <class>
    '''
    
    Returns the prompt string.
    """    
    
    for (data, targ) in d:
        prompt += 'SEQUENCE: ' + seq_enc(data) + '\n'
        prompt += f'CLASS: {targ}\n'
    return prompt

def unsupervised_prompt(d: Paired, prompt: str = '') -> str:    
    """
    Generate an unsupervised prompt of the form:
        
    '''
    0: <sequence>
    1: <sequence>
    
    n: <sequence>    
    '''
    
    Returns the prompt string.
    """    

    for index, (data,_) in enumerate(d):                
        prompt += f'{index}: ' + ','.join([f'{x:.2f}' for x in data])+'\n'        
    return prompt


def group(data: Paired) -> dict[int, list]:
    """ Takes a Paired type and converts it into a grouped dictionary """
    
    vals, targs = list(zip(*data))
    grouped = {key: [value for value, group in zip(vals, targs) if group == key] for key in set(targs)}      
    return grouped

def plot_group(grouped: dict[int, list]) -> None:
    """ Plot examples in a grouped dict, coloured by class labels"""
    
    color_list = ['red','blue','green']
    fig, ax = plt.subplots()
    for name, group in grouped.items():            
        lines = ax.plot(np.array(group).transpose(), label=None, c=color_list[name]) 
        lines[0].set_label(f'CLASS: {name}')
    plt.grid()
    plt.legend()
    
    
def load_data(file_to_load: str) -> tuple[np.ndarray, ...]:
    """ load example data """
    
    data  = np.load(file_to_load)
    targs = data[0:600,-1]    
    clean = data[0:600,0:128]    
    noisy = data[0:600,128:256]
    unseen = data[600:,0:256]    
    return clean, noisy, targs.astype('int8'), unseen

def get_data(train_samples:int=5, test_samples:int=2, pnts:int=64, steps:int=2, shuffle:bool=True) -> tuple[Paired,Paired]:
    """
    Generates a set of train and test data in the form of paired lists. 
    This function does not generate new data but rather draws a subset of 
    examples from the vae_test.npy "repository".

    Parameters
    ----------
    train_samples : int, optional
        Number of training samples. The default is 5.
    test_samples : int, optional
        Number of test samples. The default is 2.
    pnts : int, optional
        Upper count of points to use. The default is 64.
    steps : int, optional
        Number of points to skip. The default is 2.
    shuffle : bool, optional
        Shuffle the samples. The default is True.

    Returns
    -------
    tuple[Paired,Paired]
        The train and test lists with curves and associated class targets.

    """
    
    clean, noisy, targs, unseen = load_data('../vae_test.npy')
    grouped = {key: [value[0:pnts:steps] for value, group in zip(noisy, targs) if group == key] for key in set(targs)}    
    
    train_list = []
    test_list = []    
    for k, v in grouped.items():
        train_list.extend(list(zip( random.sample(v, train_samples), [k]*train_samples)))    
        test_list.extend(list(zip( random.sample(v, test_samples), [k]*test_samples)))   
        
    if shuffle:
        random.shuffle(train_list)
        random.shuffle(test_list)
        
    return train_list, test_list
