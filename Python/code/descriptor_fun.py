# -*- coding: utf-8 -*-
"""
Created on Fri May 16 13:45:03 2025

@author: BlankAdventure
"""

"""
In this example we have an existing descriptor (MaxValue) that we perhaps 
are not free to access/modify. Additionally, we have a class (Demo) that 
invokes the descriptor, but does so using a fixed/design-time value. The idea 
is to explore a few ways that we can dynamically set the class descriptors' 
value at runtime instead (i.e., dynamically, per instance).

All of the following methods produce the desired behaviour, but with varying
degrees of readability.
"""

# A simple descriptor for enforcing a maximum value 
class MaxValue():        
    def __init__(self,max_val):
        self.max_val = max_val
        
    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, obj, value):
        if value > self.max_val:
                raise ValueError(f"{self.name} must be less than {self.max_val}")
        obj.__dict__[self.name] = value  

# ALL instances of Demo will have MaxValue fixed to 5. But what if we want 
# to set this dynamically instead? See subsequent code for examples of this.
class Demo():    
    x = MaxValue(5)
    def __init__(self, x):
        self.x = x
        
        
# Creates a new class and assigns to self
class WorkingDemo1:
    def __init__(self, x, max_val):
        self.__class__ = type('ConfiguredDemo', (WorkingDemo1,), 
                              {'x': MaxValue(max_val)})
        self.x = x        
        
# Set the class attributes during init
class WorkingDemo2():
    def __init__(self, x, max_val):
        cls = self.__class__
        setattr(cls, 'x', MaxValue(max_val) )        
        vars(cls)['x'].__set_name__(cls, 'x')        
        self.x = x


# Putting the class def into a function is probably the cleanest
def WorkingDemo3(x, max_val):
    class Demo():    
        x = MaxValue(max_val)
        def __init__(self, x):
            self.x = x
    return Demo(x)


# Using inheritance (kind of goofy)
class Base(MaxValue):
    def __init__(self): 
        # We specifically DO NOT want MaxValue init called until we're ready
        # so we override it with an empty method.
        pass 
    def set_value(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class WorkingDemo4():
    x = Base()
    def __init__(self, x, max_value):
        self.x.set_value(max_value)
        self.x = x


# Essentially identical WD2, but we push everything into __new__.
class WorkingDemo5:
    def __new__(cls, x, max_val):
        setattr(cls, 'x', MaxValue(max_val) )
        vars(cls)['x'].__set_name__(cls, 'x')        
        # or equivalently:
        #cls.x = MaxValue(max_val)
        #cls.x.__set_name__(cls, 'x')        
        return super().__new__(cls)
        
    def __init__(self, x, max_val):
        self.x = x

#test1 = WorkingDemo(1,5) #max value of 5
#test2 = WorkingDemo(5,10) #max value of 10        