# -*- coding: utf-8 -*-
"""
Example of autoencoder for denoising a time series

-> This is a regression model. BCE can also be used with some modifications

"""

from synthdata import trace_clusters
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Conv1D, Conv1DTranspose, Dense, Flatten, Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.constraints import max_norm
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.backend import clear_session

def do_plots (plots, clean, noisy, preds):
    if isinstance(plots,int):
        plot_list = range(0, plots)
    else:
        plot_list = plots
    nplots = len(plot_list)
    with plt.style.context(('seaborn-v0_8-whitegrid')):
        for count, ele in enumerate(plot_list):    
            plt.subplot(nplots,1,count+1)
            plt.plot( noisy[ele,:],label='noisy' )
            plt.plot( preds[ele,:],label='pred' )
            plt.plot( clean[ele,:],label='clean' )
            plt.legend(loc='lower left')            
    plt.show()    


# Setup our synthetic signal
n_samps = 50
n_clusters = 3
n_terms = 2
pnts = 128
ns_lvl = 0.20
sf = 0.1

clean, noisy, targs = trace_clusters(n_samps, n_clusters=n_clusters, 
                                     n_terms=n_terms, pnts=pnts, 
                                     ns_lvl=ns_lvl, sf=sf, mode='flat',
                                     plot=True)

# Apply scaling
scaler = MinMaxScaler((-1,1))

# IMPORTANT: We fit our scaler based on our NOISY data, and apply it to both.
# That is, we do not apply the transform separately on the noisy, clean datasets.
# A more realistic approach would be to fit based on the noisy training data, 
# then apply to the test data.
#
# The scaler is intended to operate on each column where a column is a feature.
# We want to run it along each trace, so we transpose it on input, then (un)-
# transpose it at the end.

noisy_scaler = scaler.fit(noisy.transpose())
clean = noisy_scaler.transform(clean.transpose()).transpose()
noisy = noisy_scaler.transform(noisy.transpose()).transpose()

# Reshape to <samples=150, values=128, channles=1>
clean = clean.reshape((clean.shape[0], clean.shape[1], 1))
noisy = noisy.reshape((noisy.shape[0], noisy.shape[1], 1))

train_ratio = 0.70
train_clean, test_clean, train_noisy, test_noisy = train_test_split(clean, noisy, test_size=1-train_ratio, shuffle=True)
#%%

clear_session()

input = layers.Input(shape=(128, 1))
max_norm_value = 2


# Encoder 128/64/32 8/5/2
x = Conv1D(128, kernel_size=8, kernel_constraint=max_norm(max_norm_value), activation='gelu', kernel_initializer='he_uniform')(input)
x = Conv1D(64, kernel_size=5, kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform')(x)
x = Conv1D(32, kernel_size=2, kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform')(x)

# Decoder
x = Conv1DTranspose(32, kernel_size=2, kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform')(x)
x = Conv1DTranspose(64, kernel_size=5, kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform')(x)
x = Conv1DTranspose(128, kernel_size=8, kernel_constraint=max_norm(max_norm_value), activation='gelu', kernel_initializer='he_uniform')(x)
x = Conv1D(1, kernel_size=16, kernel_constraint=max_norm(max_norm_value), activation='tanh', padding='same')(x)

autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

autoencoder.summary()

checkpointer = ModelCheckpoint('sample.keras', verbose=True, save_best_only=True)
early_stopper = EarlyStopping(monitor='val_loss', mode='min', verbose=True, patience=8, restore_best_weights=True)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=1e-8, patience=5, verbose=True)
tensorboard_callback = TensorBoard(log_dir='c://temp//tf', histogram_freq=1)

callbacks = [reduce_lr_callback, checkpointer, early_stopper, tensorboard_callback]


history = autoencoder.fit(
    x=train_noisy,
    y=train_clean,
    epochs=200,
    batch_size=32, #32, 64 seems to work best
    shuffle=False,
    validation_split=0.2,
    verbose=2,
    callbacks=callbacks
)


preds = autoencoder.predict( test_noisy )

do_plots([3,20], test_clean, test_noisy, preds)


with plt.style.context(('seaborn-v0_8-whitegrid')):
    plt.plot(history.history['loss'],label='training loss')  
    plt.plot(history.history['val_loss'],label='val loss')
    plt.legend(loc='upper right')
plt.show()