# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 20:08:36 2025

https://medium.com/logicai/non-negative-matrix-factorization-for-recommendation-systems-985ca8d5c16c
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

V = np.array([[0,1,0,1,2,2],
              [2,3,1,1,2,2],
              [1,1,1,0,1,1],
              [0,2,3,4,1,1],
              [0,0,0,0,1,0]])

V = pd.DataFrame(V, columns=['John', 'Alice', 'Mary', 'Greg', 'Peter', 'Jennifer'])
V.index = ['Vegetables', 'Fruits', 'Sweets', 'Bread', 'Coffee']

nmf = NMF(3)
nmf.fit(V)

H = pd.DataFrame(np.round(nmf.components_,2), columns=V.columns)
H.index = ['Fruits pickers', 'Bread eaters',  'Veggies']

W = pd.DataFrame(np.round(nmf.transform(V),2), columns=H.index)
W.index = V.index

reconstructed = pd.DataFrame(np.round(np.dot(W,H),2), columns=V.columns)
reconstructed.index = V.index