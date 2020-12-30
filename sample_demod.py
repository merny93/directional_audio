import wave
from scipy.io import wavfile
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt

def convolve(ar1, ar2):
    return fft.irfft(fft.rfft(np.hamming(ar1.size)*ar1) * np.conj(fft.rfft(np.hamming(ar2.size)*ar2)))

sample_rate, data = wavfile.read("fake_data.wav")

mic_space = 0.1
speed_sound = 340
max_shift = int(np.ceil((mic_space/speed_sound)*sample_rate))
print("max shift to solve for:", max_shift)


conv = np.abs(convolve(data[:,0], data[:,1]))
print(np.argmax(conv))
plt.plot(conv)
plt.show()


dir_audio= []
dirs = np.arange(-max_shift, max_shift)
sig1_ft = fft.rfft(data[:,0])
sig2_ft = fft.rfft(data[:,1])
conv = fft.irfft(sig1_ft * np.conj(sig2_ft))
conv = conv/np.max(np.abs(conv))
for direct in dirs:
    conv[:max_shift] = 0
    conv[-max_shift:] = 0
    conv[direct] = 1
    plt.plot(conv)
    plt.show()
    conv_ft = fft.rfft(conv)
    res_audio = fft.irfft(conv_ft / np.conj(sig1_ft))
    res_audio = np.asarray(res_audio/np.max(np.abs(res_audio)) * 32767, dtype=np.int16)
    dir_audio.append(res_audio)
    wavfile.write("outputs/dir_audio" + str(direct)+ ".wav", sample_rate, res_audio)
