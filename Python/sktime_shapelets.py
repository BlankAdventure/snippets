# -*- coding: utf-8 -*-
"""
Playing with shapelets for time series classification. 
"""
import matplotlib.pyplot as plt
from synthdata import trace_clusters 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

from sktime.classification.sklearn import RotationForest
from sktime.transformations.panel.shapelet_transform import ShapeletTransform
from sktime.classification.compose import ClassifierPipeline
from sktime.classification.distance_based import ProximityTree


n_samps = 9
n_clusters = 2
_, traces, targs = trace_clusters(n_samps,
                                  plot=True,mode='flat',sf=0.05,
                                  ns_lvl=0.1,pnts=32,
                                  n_clusters=n_clusters)#,seed=1240)

# Possiby apply some sort of scaler
#traces =  MinMaxScaler().fit_transform(traces.transpose()).transpose()

traces3d = traces[:,None,:]
# This is a matrix of dimension (18, 1, 32). That's 10 curves drawn from 2
# clusters/classes (9 x 2 = 18), each with 32 time points.

# Split traces3d in to train and test datasets. Just split in half (0.5) for 
# simplicity.
x_train, x_test, y_train, y_test = train_test_split(traces3d,targs, test_size=0.5, shuffle=True)


stf = ShapeletTransform(max_shapelets_to_store_per_class=6,verbose=1, remove_self_similar=False)
res = stf.fit_transform(x_train, y_train)
# res is a matrix of dimension (9, 12) where we have 9 total examples and 
# 2 classes x 6 shapelets per class = 12.

print(res)
[print(S) for S in stf.get_shapelets()]
print(y_train[ [S.series_id for S in stf.get_shapelets()] ])

# Plot the shapelets
with plt.style.context(('seaborn-v0_8-whitegrid')):
    for S in stf.get_shapelets():
        plt.plot( S.data.flatten() )

# for RotationForest, we can pass res in directly as 2D DataFrame
clf = RotationForest(n_estimators=10)
clf.fit(res, y_train)
yp = clf.predict(stf.transform(x_test))
print(classification_report(y_test, yp))

# but for ProximityTree we have to convert to numpyflat first...
clf_ptree = ProximityTree(max_depth=5, n_stump_evaluations=5)
clf_ptree.fit(res.to_numpy(), y_train)
yp = clf_ptree.predict(stf.transform(x_test).to_numpy())
print(classification_report(y_test, yp))


# No clue why this doesn't work:
# AttributeError: 'RotationForest' object has no attribute 'clone'
# Possibly a bug in sktime?
#pipeline = ClassifierPipeline(
#     RotationForest(n_estimators=10), [ShapeletTransform(max_shapelets_to_store_per_class=6,verbose=1, remove_self_similar=False)]
#)

#pipeline.fit(x_train, y_train)
#pipeline.predict(x_test)

