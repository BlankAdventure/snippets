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


api_key = os.environ.get('genai_key')
client = genai.Client(api_key=api_key)

seq_enc = lambda v: ','.join([f'{x:.2f}' for x in v])+'\n' 

def supervised_prompt(d, prompt = ''):    
    for data, targ in zip(d['data'],d['targs']  ):
        prompt += 'SEQUENCE: ' + seq_enc(data)
        prompt += f'CLASS: {targ}\n'
    return prompt

def unsupervised_prompt(d, prompt = ''):    
    for index, data in enumerate(d['data']):                
        prompt += f'{index}: ' + ','.join([f'{x:.2f}' for x in data])+'\n'        
    return prompt


def plot_dict(d):
    cidx = ['red','blue','green']
    fig, ax = plt.subplots()
    for name, group in d.items():            
        lines = ax.plot(np.array(group).transpose(), label=None, c=cidx[name]) 
        lines[0].set_label(f'CLASS: {name}')
    plt.grid()
    plt.legend()
    
def shuffle(flat):
    paired = list (zip(flat['data'],flat['targs']))
    random.shuffle(paired)
    data, targs = zip(*paired)
    return  {'data': data, 'targs': targs}
    
def flatten(d):
    targs = []
    data = []
    for k,v in d.items():
        for e in v:
            data.append(e)
            targs.append(k)
    return {'data': data, 'targs': targs}
    
def load_data(file_to_load):
    data  = np.load(file_to_load)

    targs = data[0:600,-1]    
    clean = data[0:600,0:128]    
    noisy = data[0:600,128:256]
    unseen = data[600:,0:256]
    
    return clean, noisy, targs, unseen

train_samples = 5
test_samples = 2
pnts = 64
steps = 2

clean, noisy, targs, unseen = load_data('vae_test.npy')
targs = targs.astype('int8')
train = {key: [value[0:pnts:steps] for value, group in zip(clean, targs) if group == key][0:train_samples] for key in set(targs)}
test = {key: [value[0:pnts:steps] for value, group in zip(clean, targs) if group == key][train_samples:train_samples+test_samples] for key in set(targs)}

plot_dict(train)

train = shuffle(flatten(train))

#%% Supervised Learning Example, classification

ICL = supervised_prompt(train)

class_id = 1

prompt = 'SEQUENCE: ' + seq_enc(test[class_id][1])+'CLASS:\n'

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=ICL),
    contents=prompt
    )
    
print(f'Input class: {class_id} | Reported class: {response.text}')

#prompt = ['SEQUENCE: '+seq_enc(test[x][0])+'CLASS:\n' for x in range(3)]

#%% Unsupervised classification - json
initial = """Assign each of the following sequences to one of three clusters. 

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

#%%
# #%% Unsupervised classification 
# initial = "Assign each of the following sequences to one of three clusters:\n\n"
# ICL = unsupervised_prompt(train,initial)
# prompt = ICL + "\nCLUSTER 1:\nCLUSTER 2:\nCLUSTER 3:\n"

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     config=types.GenerateContentConfig(temperature=0.1),
#     contents=prompt
#     )
   

# print(response.text)

#%% Supervised learning example, sequence prediction
# this isn't very interesting for reasons

#prompt = '''CLASS: 0\nSEQUENCE:\n'
# prompt = '''Predict the sequence  
# CLASS: 0
# SEQUENCE:

# CLASS: 1
# SEQUENCE:
# '''

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     config=types.GenerateContentConfig(
#         system_instruction=ICL),
#     contents=prompt
#     )
    

# print(response.text)

#test = [float(s) for s in response.text.split(',')]
