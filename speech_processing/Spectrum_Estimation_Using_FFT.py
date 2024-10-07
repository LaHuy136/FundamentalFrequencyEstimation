import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal.windows import hamming
from scipy.signal import medfilt
from scipy.signal import find_peaks
import librosa
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.audio_helper import define_energy, find_voice_index

# Hàm tìm F0 bằng phương pháp HPS
def find_f0(signal, sample_rate, min_freq=60, max_freq=400, N_FFT=4096):
    # Áp dụng cửa sổ Hamming để làm mượt tín hiệu
    windowed_signal = signal * np.hamming(len(signal))
    
    # Thực hiện FFT và chỉ lấy phổ biên độ
    fft_N_points = np.fft.fft(windowed_signal, N_FFT)
    spectrum = 2.0/N_FFT * np.abs(fft_N_points[:N_FFT//2])
    frequencies_N_points = sample_rate * np.arange(N_FFT//2) / N_FFT
    
    # Giới hạn tần số trong khoảng quan tâm (60Hz - 400Hz)
    valid_freqs = (frequencies_N_points >= min_freq) & (frequencies_N_points <= max_freq)
    frequencies_N_points = frequencies_N_points[valid_freqs]
    spectrum = spectrum[valid_freqs]

    # Áp dụng HPS (Harmonic Product Spectrum)
    hps_spectrum = np.copy(spectrum)

    # Nhân phổ với các bội số 2, 3, 4,...
    for h in range(2, 10):
        # Downsample bằng cách sử dụng phép nội suy để lấy phổ tương ứng
        downsampled_spectrum = np.interp(
            np.arange(0, len(spectrum), h),  # Các giá trị sau khi nội suy
            np.arange(0, len(spectrum)),     # Các giá trị ban đầu
            spectrum                        # Phổ gốc
        )
        
        # Đảm bảo các phổ có cùng độ dài trước khi nhân
        min_len = min(len(hps_spectrum), len(downsampled_spectrum))
        hps_spectrum[:min_len] *= downsampled_spectrum[:min_len]

    # Tìm tần số có biên độ lớn nhất sau khi áp dụng HPS
    peak_freq = frequencies_N_points[np.argmax(hps_spectrum)]
    print(f'Peak frequency: {peak_freq} Hz')
    
    return peak_freq


# Hàm tính F0 cho từng frame dựa trên các index_voices
def calculate_f0_per_frame(x, index_voices, frame_length, hop_length, sr, kernel_size=5):
    f0_values = []
    time_values = []

    for idx in index_voices:
        start = idx * hop_length
        end = start + frame_length
        frame = x[start:end]

        if len(frame) == frame_length:
            frame = frame * np.hamming(frame_length)
              # Áp dụng bộ lọc trung vị
            filtered_frame = medfilt(frame, kernel_size=kernel_size)
            f0 = find_f0(filtered_frame, sr)
            f0_values.append(f0 if f0 != 0 else np.nan)
            time_values.append(librosa.frames_to_time(idx, sr=sr, hop_length=hop_length))

    return time_values, f0_values

def find_f0s(audio: np.ndarray, index_voices: list, frame_length: int, hop_length: int, sr: int) -> list:
    f0s = []  # Lưu giá trị F0 chỉ cho các frame đã xử lý

    for i in index_voices:
        start_sample_index = librosa.frames_to_samples(i, hop_length=hop_length)
        end_sample_index = start_sample_index + frame_length
        
        # Đảm bảo frame không vượt quá độ dài tín hiệu
        if end_sample_index <= len(audio): 
            frame = audio[start_sample_index:end_sample_index]
            f0 = find_f0(frame, sr, min_freq=60, max_freq=400, N_FFT=4096)
            
            # Lưu F0 vào danh sách
            f0s.append(f0)

    return f0s

def split_segments(index_voices: list) -> list:
    start = index_voices[0]
    segments = []
    for i in range(1, len(index_voices)):
        if index_voices[i] - index_voices[i-1] > 1:
            segments.append((start, index_voices[i-1]))
            start = index_voices[i]
    segments.append((start, index_voices[-1]))
    print(segments)
    return segments

def analize_audio(file_path: str, frame_length_ms, thresh=0.05) -> None:
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    frame_length = int(frame_length_ms * sr / 1000)
    hop_length = frame_length // 2
    print(f'Frame length: {frame_length}, Hop length: {hop_length}')

    # Find energy
    energy = define_energy(audio, frame_length, hop_length)
    energy_norm = energy / max(energy)
    
    t = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=hop_length)

    index_voices = find_voice_index(audio, frame_length, hop_length, thresh)

    # Find f0
    f0s = find_f0s(audio, index_voices, frame_length, hop_length, sr)
    segments = split_segments(index_voices)
    
    # Tính F0 cho các index_voices đã lọc
    time_values, f0_values = calculate_f0_per_frame(audio, index_voices, frame_length, hop_length, sr)


    # Plot waveform
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 1, 1)
    plt.plot(t, energy_norm)
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')

    plt.subplot(3, 1, 2)
    plt.scatter(time_values, f0_values, c='black', s=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # Điều chỉnh giới hạn trục y
    plt.ylim(0, 420) 

    plt.subplot(3, 1, 3)
    plt.plot(audio)
    # Đánh dấu tần số cơ bản (F0) trên đồ thị phổ tần số
    for i in segments:
        # define start and end of the segment
        start_sample_index = librosa.frames_to_samples(i[0], hop_length=hop_length)
        end_sample_index = librosa.frames_to_samples(i[1], hop_length=hop_length) + frame_length

        plt.axvline(start_sample_index, color='r')
        # fill the segment with green color
        plt.axvspan(start_sample_index, end_sample_index, color='g', alpha=0.5)
        plt.axvline(end_sample_index, color='r')
        
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.tight_layout()  
    plt.show()

if __name__ == "__main__":
    file_path = 'test_signals/FHU_RE_005.wav'
    frame_length_ms = 30
    analize_audio(file_path, frame_length_ms, 0.01)