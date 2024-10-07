import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal.windows import hamming
from scipy.signal import medfilt
from scipy.signal import find_peaks
import librosa
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.audio_helper import define_energy, find_voice_index

def split_voice_segments(signal: np.array, frame_size: int, frame_step: int, threshold: int = 0.01) -> list:
    index_voices = find_voice_index(signal, frame_size, frame_step, threshold)

    start = index_voices[0]
    segments = []
    for i in range(1, len(index_voices)):
        if index_voices[i] - index_voices[i-1] > 1:
            segments.append((start, index_voices[i-1]))
            start = index_voices[i]
    segments.append((start, index_voices[-1]))
    return segments

def calculate_f0(signal, fs, frame_size, frame_step, segments):
    """
    Hàm xác định tần số cơ bản (F0) cho từng đoạn có giọng nói sử dụng kỹ thuật Cepstrum.
    :param signal: Tín hiệu đầu vào (dạng mảng numpy)
    :param fs: Tần số lấy mẫu (sampling rate)
    :param frame_size: Kích thước của mỗi frame (tính bằng số mẫu)
    :param frame_step: Bước nhảy giữa các frame (tính bằng số mẫu)
    :param segments: Danh sách các đoạn giọng nói
    :return: Danh sách F0 cho từng frame
    """
    f0s = []

    for start, end in segments:
        # Xác định frame bắt đầu và kết thúc
        start_frame = start * frame_step
        end_frame = end * frame_step
        
        # Lặp qua từng frame trong đoạn giọng nói
        for i in range(start_frame // frame_step, (end_frame // frame_step) + 1):
            start_index = i * frame_step
            end_index = min(start_index + frame_size, len(signal))
            
            # Lấy frame hiện tại
            frame = signal[start_index:end_index]
            
            # Nếu frame ngắn hơn kích thước, thêm các giá trị 0 (zero-padding)
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
            
            # Áp dụng cửa sổ Hamming
            windowed_frame = frame * hamming(len(frame))
            
            # 2. Chuyển đổi Fourier nhanh (FFT)
            spectrum = fft(windowed_frame)
            
            # 3. Tính log phổ (Log Spectrum)
            log_spectrum = np.log(np.abs(spectrum) + np.finfo(float).eps)  # Thêm epsilon để tránh log(0)
            
            # 4. Biến đổi ngược Fourier (Inverse FFT) để thu được Cepstrum
            cepstrum = np.real(ifft(log_spectrum))
            
            # 5. Tìm đỉnh cepstrum trong khoảng quefrency tương ứng với F0
            quefrency_range = int(0.002 * fs), int(0.02 * fs)  # Khoảng 50 Hz đến 500 Hz
            weighted_cepstrum = cepstrum[quefrency_range[0]:quefrency_range[1]] * np.linspace(1, 1, quefrency_range[1] - quefrency_range[0])

            # 6. Tìm đỉnh trong khoảng quefrency liên quan đến F0
            cepstrum_peak_index = np.argmax(weighted_cepstrum) + quefrency_range[0]
            peaks, _ = find_peaks(cepstrum[quefrency_range[0]:quefrency_range[1]], height=0.1, distance=10)
            if len(peaks) > 0:
                cepstrum_peak_index = peaks[0] + quefrency_range[0]
            else:
                cepstrum_peak_index = 0
            f0 = fs / cepstrum_peak_index if cepstrum_peak_index != 0 else 0
            f0s.append(f0)

    # Gán giá trị F0 là 0 cho các frame không có giọng nói
    num_frames = int(np.ceil(float(len(signal) - frame_size) / frame_step)) + 1
    f0s_full = np.zeros(num_frames)  # Khởi tạo mảng F0 với giá trị 0
    for idx, (start, end) in enumerate(segments):
        for i in range(start * frame_step // frame_step, (end * frame_step // frame_step) + 1):
            f0s_full[i] = f0s.pop(0)  # Gán giá trị F0 vào các frame có giọng nói
    
    return f0s_full

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đọc tín hiệu từ file .wav
    file_path = 'test_signals/MDU_RE_005.wav'
    threshold = 0.005
    signal, fs = librosa.load(file_path, sr=None)
    
    # Thiết lập kích thước frame và bước nhảy
    frame_size = int(0.03 * fs)  # Frame dài 30 ms
    frame_step = int(0.015 * fs)  # Bước nhảy giữa các frame là 10 ms
    frame_length = len(signal) // frame_step + 1

    print(f"Tần số lấy mẫu: {fs} Hz")
    print(f"Số lượng khung: {frame_length} frames")

    # Tách các đoạn giọng nói
    segments = split_voice_segments(signal, frame_size, frame_step, threshold)

    for i, (start, end) in enumerate(segments):
        print(f"Đoạn giọng nói {i+1}: {start * frame_step} - {end * frame_step}")

    print(f"Số đoạn giọng nói: {len(segments)}")

    f0s = calculate_f0(signal, fs, frame_size, frame_step, segments)
    
    # Vẽ biểu đồ F0 theo thời gian (từng frame)
    plt.figure( figsize=(14, 7))
    plt.suptitle("Phân tích F0 bằng phương pháp Cepstrum: {0}".format(file_path))
    plt.subplot(2, 1, 1)
    # plot signal and segments
    plt.plot(signal)
    for start, end in segments:
        plt.axvline(start * frame_step, color='g')
        plt.axvspan(start * frame_step, end * frame_step, color='g', alpha=0.4)
        plt.axvline(end * frame_step, color='g')
    plt.xticks(np.arange(0, len(signal), fs), np.arange(0, len(signal) // fs + 1))
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Amplitude")
    plt.title("Tín hiệu âm thanh và các đoạn giọng nói")
    plt.grid(True)

    # space between two subplots
    plt.subplots_adjust(hspace=1)

    plt.subplot(2, 1, 2)
    times = np.arange(len(f0s)) * (frame_step / fs)  # Thời gian của mỗi frame
    f0s_non_zero = np.where(f0s == 0, np.nan, f0s)
    plt.scatter(times, f0s, c='black', s=2)
    plt.xticks(np.arange(0, times[-1], 1))
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Tần số cơ bản F0 (Hz)")
    plt.title("Biểu đồ F0 theo thời gian")
    plt.grid(True)
    plt.show()
