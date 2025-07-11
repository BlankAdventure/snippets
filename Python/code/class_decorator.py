# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 17:44:22 2025

@author: BlankAdventure

A decorator function that takes arguments and a class, and returns a modified class.
"""

def my_class_decorator(arg1, arg2):
    """
    This is the outer function that accepts arguments for the decorator.
    It returns the actual class decorator.
    """
    def decorator(cls):
        """
        This is the actual class decorator. It receives the class to be decorated.
        """
        print(f"Decorating class {cls.__name__} with arguments: {arg1}, {arg2}")

        # You can modify the class here, e.g., add new attributes or methods
        setattr(cls, 'decorator_arg1', arg1)
        setattr(cls, 'decorator_arg2', arg2)

        # You can also wrap existing methods if needed
        original_init = cls.__init__
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            print(f"__init__ called for {self.__class__.__name__}. Decorator args: {self.decorator_arg1}, {self.decorator_arg2}")
        cls.__init__ = new_init

        return cls  # Return the modified class
    return decorator

@my_class_decorator("value_for_arg1", 123)
class MyClass:
    def __init__(self, name):
        self.name = name
        print(f"MyClass instance created with name: {self.name}")

    def greet(self):
        print(f"Hello from {self.name}! Decorator arg1: {self.decorator_arg1}")

# Create an instance of the decorated class
obj = MyClass("Alice")
obj.greet()