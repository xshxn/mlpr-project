import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


file_path = 'static/0147.wav'
y, sr = librosa.load(file_path)

plt.figure(figsize=(12, 8))


plt.subplot(2, 1, 1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Regular Spectrogram')


plt.subplot(2, 1, 2)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.amplitude_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')

plt.tight_layout()
plt.show()