# -*- coding: utf-8 -*-
"""
From the discussion here:
https://old.reddit.com/r/DSP/comments/1fcvyu0/compute_spectrogram_phase_with_lws_locally/
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

n_fft = 2048 
hop_length = 512
sr = 22050 

# ------ GENERATE SPECTROGRAM IMG --------
y, sr = librosa.load(librosa.ex('trumpet'), sr=None)
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))

fig, ax = plt.subplots(figsize=(5.12, 5.12))
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time', ax=ax)
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
plt.savefig('output.jpeg', bbox_inches='tight', pad_inches=0, dpi=100)
# ----- END ---------

# ------ REGENERATE AUDIO FROM IMG --------
img = Image.open('output.jpeg').convert('L') #greyscale img of spectrogram 

spectrogram_array = np.array(img)
spectrogram_db = (spectrogram_array  / 255.0) * 80.0 - 80.0
spectrogram_amplitude= librosa.db_to_amplitude(spectrogram_db)

padding = max(0, (n_fft//2 + 1) - spectrogram_amplitude.shape[0])
spectrogram_amplitude = np.pad(spectrogram_amplitude, ((0, padding), (0, 0)), mode='constant')

griflim = np.abs(librosa.griffinlim(spectrogram_amplitude, n_iter=50, hop_length=hop_length//2, win_length=n_fft))
griflim = librosa.util.normalize(griflim) #without normalising there is no waveform
# ------ END --------
# Plot the waveforms
fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
librosa.display.waveshow(y, sr=sr, color='b', ax=ax[0])
librosa.display.waveshow(griflim, sr=sr, color='g', ax=ax[1])
