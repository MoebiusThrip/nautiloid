# import reload
from importlib import reload

# import math
from math import exp
import numpy

# import scipy to read wavfiles
from scipy.io import wavfile

# import matplotlib for plotting
from matplotlib import pyplot

# read in the file
frequency, data = wavfile.read('gems/Lichenthrope.wav')

# extract data
times = list(range(len(data)))
mono = [float(stereo[0]) for stereo in data]
duo = [float(stereo[1]) for stereo in data]

# set start and finish
start = 0
finish = 200000

# plot
pyplot.clf()
pyplot.plot(times[start:finish], mono[start:finish], 'b--')
pyplot.plot(times[start:finish], duo[start:finish], 'g--')
pyplot.savefig('plots/testo.png')

# apply fade
chunk = int(0.1 * len(data))
denouement = data[:chunk]

# generate fade
fade = [exp(-(index ** 2) / (chunk ** 2)) for index, _ in enumerate(denouement)]
fade = [[value, value] for value in fade]
denouement = denouement * fade
denouement = denouement.astype('int16')

# concatenate
combo = numpy.concatenate([data, denouement], axis=0)

# rewrite file
wavfile.write('gems/Lichenthrope_fade.wav', frequency, combo)