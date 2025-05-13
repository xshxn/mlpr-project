import pandas as pd 
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np



samplerate1, data1 = read('static/0147.wav')
samplerate2, data2 = read('static/no_silence_0147.wav')

#idk whats happening to 

duration1 = len(data1)/samplerate1
duration2 = len(data2)/samplerate2

time1 = np.arange(0, duration1, 1/samplerate1)
time2 = np.arange(0, duration2, 1/samplerate2)


plt.figure(figsize=(15, 5))

plt.subplot(2, 1, 1)
plt.plot(time1, data1)
plt.title(f'Audio File')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(time2, data2)
plt.title(f'Silence removed Audio File ')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')



plt.tight_layout()
plt.savefig('147_1.png')
plt.show()