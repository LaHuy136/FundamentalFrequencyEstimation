import librosa
import numpy as np
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Load an audio file (replace with your own file path)
audio_path = 'data/i.wav'
y, sr = librosa.load(audio_path, sr=None)

# convert to mono
y = librosa.to_mono(y)

# Use librosa's piptrack (pitch tracking)
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

# Select the pitch with the highest magnitude for each frame
def extract_pitch(pitches, magnitudes):
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_values.append(pitch if pitch > 0 else np.nan)
    return np.array(pitch_values)

f0 = extract_pitch(pitches, magnitudes)

# plot the pitch contour
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(f0)
plt.xlabel('Time (frames)')
plt.ylabel('Frequency (Hz)')
plt.title('Pitch contour')
plt.show()

# Clean F0 values
f0_clean = f0[~np.isnan(f0)]  # Remove NaN (unvoiced regions)
