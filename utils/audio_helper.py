import numpy as np

def define_energy(x: np.ndarray, frame_length, hop_length) -> np.ndarray:
    energy = np.array([
        sum(abs(x[i:i+frame_length]**2))
        for i in range(0, len(x), hop_length)
    ])

    return energy

def find_voice_index(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512, thresh=0.01) -> list:
    """
    Tìm index của các frame có năng lượng lớn hơn ngưỡng thresh
    - param audio: mảng âm thanh
    - param frame_length: độ dài frame
    - param hop_length: khoảng cách giữa các frame
    - thresh: ngưỡng năng lượng

    :return: list các index của các frame có năng lượng lớn hơn ngưỡng
    """
    index_voices = []

    # Tính energy
    energy = define_energy(audio, frame_length, hop_length)

    energy_norm = energy / max(energy)

    # Tạo danh sách các frame index mà energy >= ngưỡng
    for i in range(len(energy)):
        if energy_norm[i] >= thresh:
            index_voices.append(i)

    return index_voices