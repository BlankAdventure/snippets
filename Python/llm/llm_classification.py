# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:48:43 2025

@author: BlankAdventure
"""
import os
import random
import json
import pprint
from google import genai
from google.genai import types
import llm_utils
from llm_utils import supervised_prompt, unsupervised_prompt, plot_group

api_key = os.environ.get('genai_key')
client = genai.Client(api_key=api_key)


train, test = llm_utils.get_data()
plot_group(llm_utils.group(train))



#%% ********** Supervised Learning Example - Classification **********



ICL = supervised_prompt(train)

# We'll build a prompt asking for two unseen examples from the test set
examples = random.sample(test, 2)

prompt = """
Do not repeat the sequence. Only print the class.

SEQUENCE: {}
CLASS: 
SEQUNECE: {}
CLASS:
""".format(*[llm_utils.seq_enc(x[0]) for x in examples])

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=ICL,
        temperature=0.2),
    contents=prompt
    )
    
print(f'Input class: {[x[1] for x in examples]}\nReported class: [{", ".join(response.text.splitlines())}]')



#%% ********** Unsupervised Learning Example - Clustering Part 1 **********
# In this example we ask for a JSON response format in the prompt.

initial = """Assign each of the following sequences to 1 of 3 clusters. 

Use the following JSON schema in your response: 
{'cluster': int, 'sequences': list[int], 'explanation': str}

"""

prompt = unsupervised_prompt(train, initial)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(temperature=0.3,
    response_mime_type="application/json"),        
    contents=prompt
    )
   

json_dict = json.loads(response.text)
pprint.pprint(json_dict)

from colorama import Fore

# Convert the json dict into a grouped dict for plotting, and print the 
# explanations in colors.
ai_grouped = {}
clrs = [Fore.RED, Fore.BLUE, Fore.GREEN]
for entry in json_dict:    
    this_class = entry['cluster']
    sequences = entry['sequences']
    ai_grouped[this_class] = [ train[x][0] for x in sequences]    
    print(clrs[this_class] + f'\nCLASS {this_class}:\n' + entry['explanation'])

plot_group(ai_grouped)

#%% ********** Unsupervised Learning Example - Clustering Part 2 **********
# In this example we supply our JSON schema directly to the model.


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
    contents=prompt
    )

print(response.parsed) 
