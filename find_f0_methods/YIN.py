import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Step 1: Load an audio file
audio_path = 'data/i.wav'  # Replace with the path to your audio file
y, sr = librosa.load(audio_path)

# Step 2: Remove silent frames
y, _ = librosa.effects.trim(y)

# Step 3: Apply YIN algorithm for F0 estimation
f0_values = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Step 4: Create a time axis for plotting
times = librosa.times_like(f0_values, sr=sr)

# Step 4: Plot the F0 estimation
plt.figure(figsize=(14, 5))
plt.plot(times, f0_values, label='Estimated F0 (Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Fundamental Frequency Estimation using YIN Algorithm')
plt.grid()
plt.legend()
plt.show()
