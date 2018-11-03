from pydub import AudioSegment
import numpy as np
# from PIL import Image

sound = AudioSegment.from_mp3("voyager-image-audio.mp3")

# get raw audio data as a bytestring
raw_data = sound.raw_data
# get the frame rate
sample_rate = sound.frame_rate
# get amount of bytes contained in one sample
sample_size = sound.sample_width
# get channels
channels = sound.channels
print(channels)
print(sample_rate)
print(sample_size)
arr = np.fromstring(raw_data, dtype=np.int16)
channelOne = arr[::2]
channelTwo = arr[1::2]
print(len(arr))
print(len(channelOne))
print(len(channelTwo))


from matplotlib import pyplot as plt
plt.figure(1)
plt.title('Signal Wave...')
plt.plot(channelOne[10000:50099])
plt.show()