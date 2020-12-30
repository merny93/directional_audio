import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index = 0,
                frames_per_buffer=CHUNK)


print("* recording")

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()


wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

exit()

##create the workable signal
signal = np.frombuffer(b''.join(frames), "Int16")
#Split the data into channels 
channels = [[] for channel in range(CHANNELS)]
for index, datum in enumerate(signal):
    channels[index%len(channels)].append(datum)
channels = [np.array(x, dtype=np.int16) for x in channels]
tbase=np.linspace(0, len(signal)/len(channels)/RATE, num=int(len(signal)/len(channels)))
plt.clf()
plt.figure(1)
plt.title('Signal Wave...')
for channel in channels:
    plt.plot(tbase,channel)
plt.savefig("sample.png")
plt.show()

import scipy as sp

conv = sp.fft.irfft(sp.fft.rfft(np.hanning(channels[0].size) *channels[0]) *np.conj(sp.fft.rfft(np.hanning(channels[1].size) * channels[1])))
print(np.argmax(conv))
plt.clf()
plt.plot(conv)
plt.show()

wf = wave.open("one_chanel.wav", 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(channel[1])
wf.close()

