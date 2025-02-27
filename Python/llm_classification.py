# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:48:43 2025

@author: BlankAdventure
"""
import os
import random
import json
import pprint
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from typing import TypedDict



api_key = os.environ.get('genai_key')
client = genai.Client(api_key=api_key)

# Helper function for stringifying a list of floats
seq_enc = lambda v: ','.join([f'{x:.2f}' for x in v])+'\n' 



class Flat(TypedDict):
    """ Flat-style dictionary for holding curves and their labels """
    
    data: list|tuple
    targs: list|tuple
    

def supervised_prompt(d: Flat, prompt: str = '') -> str:
    """
    Generate a supervised prompt of the form:
        
    '''
    SEQUENCE: <sequence>
    CLASS: <class>
    '''
    
    Returns the prompt string.
    """    
    
    for data, targ in zip(d['data'],d['targs']  ):
        prompt += 'SEQUENCE: ' + seq_enc(data)
        prompt += f'CLASS: {targ}\n'
    return prompt

def unsupervised_prompt(d: Flat, prompt: str = '') -> str:    
    """
    Generate an unsupervised prompt of the form:
        
    '''
    0: <sequence>
    1: <sequence>
    
    n: <sequence>    
    '''
    
    Returns the prompt string.
    """    

    for index, data in enumerate(d['data']):                
        prompt += f'{index}: ' + ','.join([f'{x:.2f}' for x in data])+'\n'        
    return prompt


def plot_dict(grouped: dict[int, list]) -> None:
    """ Plot examples in a grouped dict, coloured by class labels"""
    
    color_list = ['red','blue','green']
    fig, ax = plt.subplots()
    for name, group in grouped.items():            
        lines = ax.plot(np.array(group).transpose(), label=None, c=color_list[name]) 
        lines[0].set_label(f'CLASS: {name}')
    plt.grid()
    plt.legend()
    
def shuffle(flat: Flat) -> Flat:
    """ Randomly shuffles the examples in a Flat dict """
    
    paired = list (zip(flat['data'],flat['targs']))
    random.shuffle(paired)
    data, targs = zip(*paired)
    return {'data': data, 'targs': targs}

def flatten(grouped: dict[int, list]) -> Flat:
    """ Converts a grouped dict into a flattened (Flat) dict """
    
    targs = []
    data = []
    for k,v in grouped.items():
        for e in v:
            data.append(e)
            targs.append(k)
    return {'data': data, 'targs': targs}
    
def load_data(file_to_load: str) -> tuple[np.ndarray, ...]:
    """ load example data """
    
    data  = np.load(file_to_load)
    targs = data[0:600,-1]    
    clean = data[0:600,0:128]    
    noisy = data[0:600,128:256]
    unseen = data[600:,0:256]    
    return clean, noisy, targs.astype('int8'), unseen

# Used to reduce the total length of the sequences (128)
pnts = 64
steps = 2

# Desired examples per class (3)
train_samples = 5 
test_samples = 2

clean, noisy, targs, unseen = load_data('vae_test.npy')
train_grouped = {key: [value[0:pnts:steps] for value, group in zip(noisy, targs) if group == key][0:train_samples] for key in set(targs)}
test_grouped = {key: [value[0:pnts:steps] for value, group in zip(noisy, targs) if group == key][train_samples:train_samples+test_samples] for key in set(targs)}

plot_dict(train_grouped)

train = shuffle(flatten(train_grouped))

#%% Supervised Learning Example, classification

ICL = supervised_prompt(train)

class_id = 1

prompt = 'SEQUENCE: ' + seq_enc(test_grouped[class_id][1])+'CLASS:\n'

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=ICL),
    contents=prompt
    )
    
print(f'Input class: {class_id} | Reported class: {response.text}')

#prompt = ['SEQUENCE: '+seq_enc(test[x][0])+'CLASS:\n' for x in range(3)]

#%% Unsupervised classification - json
initial = """Assign each of the following sequences to 1 of 3 clusters. 

Use the following JSON schema in your response: 
{'cluster': int, 'sequences': list[int], 'explanation': str}

"""

prompt = unsupervised_prompt(train)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(temperature=0.3,
    response_mime_type="application/json"),        
    contents=[initial, prompt]
    )
   

json_dict = json.loads(response.text)
pprint.pprint(json_dict)

from colorama import Fore

# Convert the json dict into a grouped dict for plotting, and print the 
# explanations in colors.
ai_groups = {}
clrs = [Fore.RED, Fore.BLUE, Fore.GREEN]
for entry in json_dict:    
    this_class = entry['cluster']
    sequences = entry['sequences']
    ai_groups[this_class] = [ train['data'][x] for x in sequences]    
    print(clrs[this_class] + f'\nCLASS {this_class}:\n' + entry['explanation'])

plot_dict(ai_groups)

#%% Try with json schema
import typing_extensions as typing

class BaseType(typing.TypedDict):
    cluster: int
    sequences: list[int]
    explanation: str


initial = """You are a machine learning clustering algorithm. 
Assign each of the following sequences to 1 of 3 clusters:
    
"""

prompt = unsupervised_prompt(train, initial)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config={'response_mime_type': 'application/json',
            'response_schema': list[BaseType],
            'temperature': 0.3
            },
    contents=[prompt]
    )

print(response.parsed) 
