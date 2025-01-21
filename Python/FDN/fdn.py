# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:47:39 2025

@author: BlankAdventure
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import hadamard

def gaussian(L:int , s:float) -> NDArray:
    x = np.arange(-L//2,L//2)
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )


def get_matrix(N:int, name:str='identity', **kwargs:Any)-> NDArray:
    '''
    Helper function to generate a variety of scattering (or feedback) 
    matrices.

    Parameters
    ----------
    N : int
        Number of delay elements (matrix will be N x N)
    name : str, optional
        Name of matrix to create. The default is 'identity'.

    Returns
    -------
    NDArray
        N x N matrix of indicated type.

    '''
    match name:
        case 'fixeddecay':
            return np.ones((N,N))*kwargs['decay']
        case 'uniform':
            return np.ones( (N,N) ) / N
        case 'random':
            return np.random.uniform(-1,1,size=(N,N))
        case 'randmap':
            return np.random.permutation( np.identity(N) )
        case 'hadamard':
            return hadamard(N)/2
        case ('identity' | _):
            return np.identity(N)
            
def input_seq(N:int, name:str='impulse', **kwargs: Any) -> NDArray:
    '''
    Generates a variety of discrete input sequences

    Parameters
    ----------
    N : int
        length of input sequence.
    name : str, optional
        Name of sequence type. The default is 'impulse'.
    **kwargs : any
        Depends on input type. See code for details.

    Returns
    -------
    x : vector
        The generated input sequence

    '''
    match name:
        case 'pulse':
            g = gaussian(kwargs['l'], kwargs['s'])
            x = np.zeros(N)
            x[0:kwargs['l']] = g
        case 'ping':
            x = np.zeros(N)
            x[0::kwargs['freq']] = 1
        case 'sine':
            x = np.sin(2*np.pi*np.arange(N)*kwargs['freq']/N)
        case 'noise':
            x = np.random.randn((N))
        case 'impulse':
            x = np.zeros(N)
            x[0] = 1            
    return x            

class Dline():
    '''
    Models a simple delay line (linear FIFO buffer)
    '''
    def __init__(self, length:int, init:float=0):
        '''
        Parameters
        ----------
        length : int
            length of delay line in number of samples
        init : float, optional
            Optional inital value for the start of
            the buffer. The default is 0.

        Returns
        -------
        None.

        '''
        self.length = length
        self.init = init
        self.buffer = np.zeros( (self.length+0))
        self.buffer[0] = init        
        
    def shift(self, in_val:float) -> float:
        '''        
        Shit the buffer from left to right by one unit. Returns
        the right-most buffer value.

        Parameters
        ----------
        in_val : float
            Optional new value to shift in.

        Returns
        -------
        float
            Right-most buffer value after applying shift.

        '''
        self.buffer = np.roll(self.buffer,1,axis=0)
        self.buffer[0] = in_val    
        return self.buffer[-1]

class FDN():
    '''
    Class for modeling a feedback delay network.
    '''
    def __init__(self, 
                 delay_list:list[int], 
                 scattering_mat: NDArray|None,
                 pass_through:bool = False):
        '''
        

        Parameters
        ----------
        delay_list : list[int]
            List of delays. Each delay defines a corresponding delay line 
            in the model. 
        scattering_mat : NDArray|None
            Scattering (or feedback) matrix.
        pass_through : bool|None
            Include no-delay path - passes input value directly to output.

        Returns
        -------
        None.

        '''
        self.delays = delay_list.copy()
        self.paths = []
        self.feedback = np.zeros( (len(delay_list)) )
        self.pass_through = pass_through
        self.smat = (get_matrix(len(delay_list),'identity') if scattering_mat 
                     is None else scattering_mat)
        
        for d in delay_list:
            self.paths.append(Dline(d))
            
    def tick(self, in_val:float) -> float:
        '''
        Simulates a single clock tick

        Parameters
        ----------
        in_val : float
            New value to feed into FDN.

        Returns
        -------
        float
            Output of FDN after applying tick.

        '''
       
        out = [de.shift(fb+in_val) for de, fb in zip(self.paths, self.feedback)]
        self.feedback = np.matmul(out,self.smat)  
        if self.pass_through:
            out.append(in_val)
        return sum(out)
