import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Đọc tín hiệu âm thanh
filename = 'data/i.wav'  # Thay đổi đường dẫn
y, sr = librosa.load(filename, sr=None)

# convert to mono
y = librosa.to_mono(y)

# Tiền xử lý: Lọc và chuẩn hóa
y = y / np.max(np.abs(y))

from scipy.signal import spectrogram

# Tính toán spectrogram
frequencies, times, Sxx = spectrogram(y, fs=sr, nperseg=1024)

# Chuyển đổi Sxx sang dB
Sxx_db = 10 * np.log10(Sxx)

# Tìm đỉnh trong mỗi cột của spectrogram
F0_indices = np.argmax(Sxx, axis=0)
F0_freqs = frequencies[F0_indices]

# Lọc tần số F0
F0 = np.mean(F0_freqs[(F0_freqs > 20) & (F0_freqs < 100)])  # Giới hạn tần số cho F0
print("F0:", F0)

plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, Sxx_db, shading='gouraud')
plt.colorbar(label='Intensity (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Cohen\'s Class Time-Frequency Distribution')
plt.ylim(0, 400)  # Giới hạn tần số để dễ nhìn hơn
plt.show()
# Output: F0: 125.0