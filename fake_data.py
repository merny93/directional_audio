import wave
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
data_files = ["byebye.wav", "hello.wav"]
data_set = [{key: wavfile.read(data_name)[i] for i,key in enumerate(["sample_rate", "data"])}for data_name in data_files]
for i in range(len(data_set)):
    data_set[i]["data"] = np.mean(data_set[i]["data"], axis = 1)

speed_sound = 360
mic_sep = 0.1
angles = [0.5, -1] #from normal in radians
delta_r = [mic_sep*np.sin(angle) for angle in angles]
delta_t = [r/speed_sound for r in delta_r]
sample_rate = data_set[0]["sample_rate"]
delta_sample = [int(np.round(sample_rate*t)) for t in delta_t]
print(delta_sample)

right_chan = []
left_chan = []
for i, offset in enumerate(delta_sample):
    right_chan.append(np.roll(data_set[i]["data"], offset))
    left_chan.append(data_set[i]["data"])
    plt.plot(right_chan[-1])
    plt.plot(left_chan[-1])
    plt.show()

right_chan = np.sum(np.array(right_chan), axis =0)
left_chan = np.sum(np.array(left_chan), axis=0)
audio_stream = np.asarray(np.concatenate((left_chan[:, np.newaxis], right_chan[:, np.newaxis]), axis = 1), dtype=np.int16)
print(np.max(audio_stream))
print(np.min(audio_stream))
print(audio_stream.dtype)
wavfile.write("fake_data.wav", sample_rate, audio_stream)