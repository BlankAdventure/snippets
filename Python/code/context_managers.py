# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 23:47:51 2025

@author: BlankAdventure
"""
from contextlib import contextmanager
import time

# As context manager:
class TimerCM:
    def __enter__(self):        
        # *** do something before ***
        self.start = time.perf_counter()
        return self #control returns to with-block

    def __exit__(self, exc_type, exc_val, exc_tb):
        # *** do something after ***
        self.stop = time.perf_counter()
        self.elapsed = self.stop - self.start
        print (f'that took {self.elapsed} seconds.')    

# As a decorator class:
class TimerDec:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        # *** do something before ***
        self.start = time.perf_counter()
        
        result = self.function(*args, **kwargs)
        
        # *** do something after ***
        self.stop = time.perf_counter()
        self.elapsed = self.stop - self.start
        print (f'that took {self.elapsed} seconds.')    
        return result

# Using the contextmanager decorator:
@contextmanager
def timer_cm():
   # *** do something before ***
   start = time.perf_counter()
   yield # returns execution to with-block
   
   # *** do something after ***
   stop = time.perf_counter()
   elapsed = stop - start
   print (f'that took {elapsed} seconds.')    

# As a decorator function:
def timer_dec(func):
    def wrapper(*args, **kwargs):
        # *** do something before ***
        start = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        # *** do something after ***
        stop = time.perf_counter()
        elapsed = stop - start
        print (f'that took {elapsed} seconds.')    
        
        return result
    return wrapper

@TimerDec 
def count1():
    result = sum(range(10_000_000))
    return result

@timer_dec
def count2():
    result = sum(range(10_000_000))
    return result


with TimerCM():
    result = sum(range(10_000_000))
    

with timer_cm():
    result = sum(range(10_000_000))


count1()    
count2()    

    
    
    