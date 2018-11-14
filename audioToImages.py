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
print("Sample Rate: %s" % rate) #384000
# print(len(data)) #181960704

# 384000 / 120 = 3200

# https://physics.stackexchange.com/questions/118647/what-is-the-unit-of-time-on-the-voyager-golden-record
# 11845632 = scan distance 101101001100000000000000
# 0.70*10^-9 = hydrogen tansition period 
# 11845632 * 0.70*10^-9s = 0.0082919424 s =~ 8.3 msec
# 44100 g=hz / (11845632 * 0.70*10^-9s) = 5318416.10477
# 44100 / 120 = 367.5


# 11845632 * 7.04024183647 * 10^(-10) = 0.00833961139 =~8.3

# multiple by -1 as images seen to be negatives
channelOne  = -1*data[:, 0]
channelTwo  = -1*data[:, 1]
image_width = 512
scan_width  = 3200

start = 6000352 #start of images
num_scans = image_width
image_data = np.zeros((3200, num_scans))

#a few image starts
start_of_image = 6000352
start_of_image = 8309753
start_of_image = 10664534
# start_of_image = 19607588
# start_of_image = 90737287
# start_of_image = 117466208

test_data = channelOne[start_of_image:]
test_image = np.zeros((1, len(test_data)))
start_of_row=[0]

i = 0 + 3000
while i <= 1658400:
    # if i %(3300 ) >= (3100 ) and i % (3100) <= (300) and i >= 6: # and i - start_of_row[-1] >= 3000:
    if all(item >= 0 for item in test_data[i-2:i+1]) and all(item <= 0 for item in test_data[i-5:i-2]):
        i = i-5
        # shift += abs(3200- (i - start_of_row[-1]))
        start_of_row.append(i)
        i += 3000
        continue
    i+=1

start_of_row = start_of_row[:image_width]

for i in range(len(start_of_row)-1):
    start = start_of_row[i]
    length_diff = scan_width - (start_of_row[i+1] - start)
    if length_diff > 0: #shorter than 3200 pixels, repeat
        subset = test_data[start:start_of_row[i+1]]
        for x in range(length_diff):
            subset = np.append(subset, subset[-x])
    else: #longer, trim
        subset = test_data[start:start+scan_width]

    # print(subset.shape)
    image_data[:,i] = np.array([subset]).T[:,0]

# print(image_data)
# image_data = image_data / np.linalg.norm(image_data)
# max_v = np.max(np.abs(image_data))

# image_data /= np.average(np.abs(image_data))
# image_data *= (255.0/image_data.max())

plt.figure(1)
plt.imshow(image_data, aspect="auto", cmap = plt.cm.gray)
# plt.imshow(image_data, aspect="auto")
# plt.imshow(image_data, cmap = plt.cm.gray)
# plt.imshow(image_data)

# plt.figure(2)
# plt.title('Signal Wave...')
# plt.plot(channelOne)
plt.show()