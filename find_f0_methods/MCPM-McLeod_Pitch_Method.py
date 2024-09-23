import numpy as np
from scipy.signal import find_peaks
from scipy.io import wavfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def is_voice(signal, threshold=0.01):
    """
    Kiểm tra xem tín hiệu có chứa giọng nói hay không.

    Parameters:
    - signal: mảng numpy, tín hiệu âm thanh mono.
    - threshold: ngưỡng để xác định tín hiệu có chứa giọng nói hay không.

    Returns:
    - True nếu tín hiệu có chứa giọng nói, ngược lại trả về False.
    """
    # Tính năng lượng tín hiệu
    energy = np.sum(signal ** 2) / len(signal)

    # Nếu năng lượng lớn hơn ngưỡng thì xem như có chứa giọng nói
    return energy > threshold

def mpm_pitch_detection(signal, fs, window_size=2048, hop_size=512, min_f0=50, max_f0=500):
    """
    Xác định F0 của tín hiệu âm thanh sử dụng Phương pháp McLeod Pitch (MPM).

    Parameters:
    - signal: mảng numpy, tín hiệu âm thanh mono.
    - fs: tần số lấy mẫu của tín hiệu.
    - window_size: kích thước cửa sổ phân tích.
    - hop_size: bước nhảy cửa sổ.
    - min_f0: tần số cơ bản tối thiểu (Hz).
    - max_f0: tần số cơ bản tối đa (Hz).

    Returns:
    - f0s: danh sách các giá trị F0 theo thời gian.
    - times: danh sách các thời điểm tương ứng với F0.
    """
    # Normalize tín hiệu
    signal = signal / np.max(np.abs(signal))

    # Số mẫu tương ứng với min và max F0
    min_lag = int(fs / max_f0)
    max_lag = int(fs / min_f0)

    f0s = []
    times = []

    num_frames = int((len(signal) - window_size) / hop_size) + 1

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + window_size]

        if not is_voice(frame):
            f0s.append(0)
            times.append(start / fs)
            continue

        # Tự tương quan
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Trích xuất đoạn tự tương quan trong khoảng lag tìm được
        autocorr = autocorr[min_lag:max_lag]

        # Tìm các cực đại trong tự tương quan
        peaks, _ = find_peaks(autocorr)

        if len(peaks) == 0:
            f0s.append(0)
            times.append(start / fs)
            continue

        # Chọn cực đại có giá trị cao nhất
        peak = peaks[np.argmax(autocorr[peaks])]

        # Ước lượng lag với nội suy tuyến tính để cải thiện độ chính xác
        if peak > 0 and peak < len(autocorr) -1:
            alpha = autocorr[peak -1]
            beta = autocorr[peak]
            gamma = autocorr[peak +1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            peak = peak + p

        # Tính F0
        f0 = fs / (peak + min_lag)
        f0s.append(f0)
        times.append(start / fs)

    return f0s, times

# Ví dụ sử dụng với file WAV
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Đọc file WAV
    fs, signal = wavfile.read('data/i.wav')

    # Nếu tín hiệu stereo, chuyển thành mono
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    # Gọi hàm MPM
    f0s, times = mpm_pitch_detection(signal, fs)

    # Vẽ đồ thị F0 theo thời gian
    plt.figure(figsize=(12, 6))
    plt.plot(times, f0s, label='F0 (Hz)')
    plt.xlabel('Thời gian (s)')
    plt.ylabel('F0 (Hz)')
    plt.title('Xác định F0 sử dụng Phương pháp McLeod Pitch (MPM)')
    plt.legend()
    plt.grid()
    plt.show()
