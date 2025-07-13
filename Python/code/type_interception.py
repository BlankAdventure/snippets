# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 14:52:42 2025

@author: BlankAdventure
"""

# 
# Idea here was to create a decorator that inspects the function arguments for 
# any Unit types and apply some modification to them automatically. 
# Kind of a clever idea that could be useful in some situations where we want
# to manipulate a certain variable type without explicitly doing so in the body
# of the function
#


import inspect
from typing import TypeAlias
Unit: TypeAlias = str|float|int

def decorator(func):    
    filtered_dict = {key: value for key, value in func.__annotations__.items() if type(value) is type(Unit)}
    def wrapper(*args, **kwargs):        
        bound_args = inspect.signature(func).bind(*args, **kwargs)        
        for key in filtered_dict:
            bound_args.arguments[key] = bound_args.arguments[key] * 2        
        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper


@decorator
def func(a: int, b: Unit, c: Unit = 0):
    print(f'function called! {a} {b} {c}')
    
func(1,2,3)

