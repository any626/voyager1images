from pydub import AudioSegment
import numpy as np
from matplotlib import pyplot as plt
# from PIL import Image


import scipy.io.wavfile
rate, data = scipy.io.wavfile.read('384kHzStereo.wav')
# rate, data = scipy.io.wavfile.read('voyager-image-audio-as-wave.wav')
# sound = AudioSegment.from_file("384kHzStereo.wav")

# get raw audio data as a bytestring
# raw_data = sound.raw_data
# print(sound)
# data = sound.get_array_of_samples()
# get the frame rate
# sample_rate = sound.frame_rate
# get amount of bytes contained in one sample
# sample_size = sound.sample_width
# get channels
# channels = sound.channels
print(rate) #384000
print(len(data)) #181960704

# 384000 / 120 = 3200

# https://physics.stackexchange.com/questions/118647/what-is-the-unit-of-time-on-the-voyager-golden-record
# 11845632 = scan distance 101101001100000000000000
# 0.70*10^-9 = hydrogen tansition period 
# 11845632 * 0.70*10^-9s = 0.0082919424 s =~ 8.3 msec
# 44100 g=hz / (11845632 * 0.70*10^-9s) = 5318416.10477
# 44100 / 120 = 367.5


# 11845632 * 7.04024183647 * 10^(-10) = 0.00833961139 =~8.3

channelOne = -1*data[:, 0] # len 41,794,099
channelTwo = data[:, 1]
image_width = 512

# print(channelOne[6000352: 6000358])
# print(np.sum(channelOne[6000352: 6000358]))

# print(channelOne[6003545: 6003551])
# print(np.sum(channelOne[6003545: 6003551]))

# 6003541 - 6000355 = 3186
# 6003545 - 6000352 = 3193
#5875
scan_width = 3200 - 3
offset = (scan_width*image_width*3 + scan_width * 350 - 500) * 1# int(scan_width*512*2)
offset = 6000000
offset = 6000352
# offset = scan_width*image_width*3 + scan_width*300# int(scan_width*512*2)
num_scans = image_width
image_data = np.zeros((scan_width, num_scans))
# image_data = np.zeros((scan_width, 400))
# print(image_data.shape)
for scan in range(num_scans):

    # if scan %2 == 1 :
    #     image_data[:,scan] = np.flip([0.5*channelOne[offset:offset+scan_width]]).T[:, 0];
    # else:
    
    image_data[:,scan] = np.array([channelOne[offset:offset+scan_width]]).T[:, 0];
    offset += scan_width

# image_data = np.array(image_data)
# print(image_data.shape)
# print(image_data[0])
# x = np.array(channelOne)
# x.reshape((-1, 512))

# print(image_data)
# image_data = image_data / np.linalg.norm(image_data)
# max_v = np.max(np.abs(image_data))

image_data /= np.average(np.abs(image_data))
image_data *= (255.0/image_data.max())

plt.figure(1)
plt.imshow(image_data, aspect="auto", cmap = plt.cm.gray)
# plt.imshow(image_data, aspect="auto")
# plt.imshow(image_data, cmap = plt.cm.gray)
# plt.imshow(image_data)

# plt.figure(2)
# plt.title('Signal Wave...')
# plt.plot(channelOne)
plt.show()