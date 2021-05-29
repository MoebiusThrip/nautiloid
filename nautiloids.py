# import reload
from importlib import reload

# import ox
import os

# import datetime
from datetime import datetime, timedelta
from time import time

# import math
from math import exp, sqrt, sin, cos, pi
import numpy

# import scipy to read wavfiles
from scipy.io import wavfile

# import matplotlib for plotting
from matplotlib import pyplot


# make class nautiloid to analyze songs with autocorrelation
class Nautiloid(list):
    """Class Nautiloid to analyze songs with color.

    Inherits from:
        list
    """

    def __init__(self, directory):
        """Initialize a Nautiloid instance with a directory.

        Arguments:
            directory: str, directory path
        """

        # set time
        self.now = time()

        # define directory of songs
        self.directory = directory

        return

    def __repr__(self):
        """Generate on-screen representation.

        Arguments:
            None

        Returns:
            str
        """

        # create representation
        representation = '< Nautiloid: {} >'.format(self.directory)

        return representation

    def _autocorrelate(self, snippet):
        """Perform autocorrelation analysis of the data.

        Arguments:
            snippet: snippet of data

        Returns:
            list of floats
        """

        # timestamp
        self._stamp('autocorrelating...', initial=True)

        # for each lag
        autocorrelation = []
        for lag, _ in enumerate(snippet):

            # create data points
            independents = snippet[:len(snippet) - lag]
            dependents = snippet[lag:]

            # compute pearson's correlation coefficient
            pearson = self._peer(independents, dependents)
            autocorrelation.append(pearson)

        # stamp
        self._stamp('autocorrelated.')

        return autocorrelation

    def _draw(self, lines, text, destination):
        """Draw a group of lines.

        Arguments:
            lines: list of tuples
            text: list of str
            destination: str, filepath

        Returns:
            None
        """

        # begin figure
        pyplot.clf()

        # collect all data points
        data = [float(entry) for line in lines for entry in line[1]]

        # calculate quartiles
        head = numpy.percentile(data, 75)
        median = numpy.percentile(data, 50)
        tail = numpy.percentile(data, 25)

        # calculate margins
        margin = max([head - median, median - tail])
        top = median + 5 * margin
        bottom = median - 5 * margin

        # set y-axis range
        pyplot.gca().set_ylim(bottom, top)

        # plot all lines
        for abscissa, ordinate, style, label in lines:

            # if there is a label
            if label:

                # plot with a label
                pyplot.plot(abscissa, ordinate, style, label=label)

            # otherwise
            else:

                # plot without a label
                pyplot.plot(abscissa, ordinate, style)

        # add legend
        pyplot.legend(loc='lower right')

        # parse text, padding with blanks
        text = text + ['', '', '']
        title, independent, dependent = text[:3]

        # add labels
        pyplot.title(title)
        pyplot.xlabel(independent)
        pyplot.ylabel(dependent)

        # save to destination
        pyplot.savefig(destination)

        return None

    def _listen(self, name):
        """Get a wave file by listening to a song.

        Arguments:
            name: partial song title

        Returns:
            None
        """

        # get all paths
        paths = self._see(self.directory)

        # get all paths with the name, assuming the first is correct
        song = [path for path in paths if name.lower() in path.lower()][0]

        # open the wave file
        frequency, data = wavfile.read(song)

        return frequency, data

    def _peer(self, independents, dependents):
        """Calculate pearson's correlation coefficient between two sets.

        Arguments:
            independents: list of floats
            dependents: list of floats

        Returns:
            float, pearson's correlation coefficient.
        """

        # default pearson to 0
        pearson = 0.0

        # check for nonzero length lists
        if len(dependents) > 0:

            # subtract mean from independents
            mean = sum(independents) / len(independents)
            independents = [entry - mean for entry in independents]

            # subtract mean from dependents
            mean = sum(dependents) / len(dependents)
            dependents = [entry - mean for entry in dependents]

            # compute the covariance
            covariance = sum([x * y for x, y in zip(independents, dependents)])

            # compute the variances
            variance = sum([x ** 2 for x in independents])
            varianceii = sum([y ** 2 for y in dependents])

            # if variances are not zero
            if variance > 0 and varianceii > 0:

                # compute pearson, coefficient: covar / sqrt(var * varii)
                pearson = covariance / sqrt(variance * varianceii)

        return pearson

    def _see(self, directory):
        """See all the paths in a directory.

        Arguments:
            directory: str, directory path

        Returns:
            list of str, the file paths.
        """

        # make paths
        paths = ['{}/{}'.format(directory, path) for path in os.listdir(directory)]

        return paths

    def _stamp(self, message, initial=False):
        """Start timing a block of code, and print results with a message.

        Arguments:
            message: str
            initial: boolean, initial message of block?

        Returns:
            None
        """

        # get final time
        final = time()

        # calculate duration and reset time
        duration = round(final - self.now, 5)
        self.now = final

        # if iniital message
        if initial:

            # add newline
            message = '\n' + message

        # if not an initial message
        if not initial:

            # print duration
            print('took {} seconds.'.format(duration))

        # begin new block
        print(message)

        return None

    def _transform(self, snippet):
        """Calculate the fourier transform.

        Arguments:
            snippet: list of ints

        Returns:
            list of floats
        """

        # get the total number of frames
        number = len(snippet)

        # calculate each point of the fourier
        fourier = []
        for wave, _ in enumerate(snippet):

            # caluculate summataion
            terms = [amplitude * cos(2 * pi * wave * index / number) for index, amplitude in enumerate(snippet)]
            fourier.append(sum(terms))

        return fourier

    def undulate(self, name, size, start=0):
        """Get the frequency spectrum for a snippet of a song.

        Arguments:
            name: (partial) name of song
            size: number of frames
            start: starting position

        Returns:
            None
        """

        # get the song file
        frequency, song = self._listen(name)
        print('frequency: {} frames / s'.format(frequency))
        print('length: {} frames, {} seconds'.format(len(song), len(song) / frequency))

        # get the first channel snippet
        snippet = song[start:start + size, 0]

        # get the fourier transform
        fourier = self._transform(snippet)

        # make spectrum from frequency
        spectrum = [frequency / (index + 1) for index, _ in enumerate(snippet)]

        # create the line
        line = (spectrum, fourier, 'b-', 'auto')
        lines = [line]

        # add plot?
        lineii = (spectrum, snippet, 'g-', 'snippet')
        lines.append(lineii)

        # add labels
        title = 'fourier transform of {}'.format(name)
        independent = 'frequency'
        dependent = 'amplitude'
        texts = [title, independent, dependent]

        # make destination
        destination = 'plots/{}.png'.format(name)

        # draw it
        self._draw(lines, texts, destination)

        return None


#
# # read in the file
# frequency, data = wavfile.read('gems/Lichenthrope.wav')
#
# # extract data
# times = list(range(len(data)))
# mono = [float(stereo[0]) for stereo in data]
# duo = [float(stereo[1]) for stereo in data]
#
# # set start and finish
# start = 0
# finish = 200000
#
# # plot
# pyplot.clf()
# pyplot.plot(times[start:finish], mono[start:finish], 'b--')
# pyplot.plot(times[start:finish], duo[start:finish], 'g--')
# pyplot.savefig('plots/testo.png')
#
# # apply fade
# chunk = int(0.1 * len(data))
# denouement = data[:chunk]
#
# # generate fade
# fade = [exp(-(index ** 2) / (chunk ** 2)) for index, _ in enumerate(denouement)]
# fade = [[value, value] for value in fade]
# denouement = denouement * fade
# denouement = denouement.astype('int16')
#
# # concatenate
# combo = numpy.concatenate([data, denouement], axis=0)
#
# # rewrite file
# wavfile.write('gems/Lichenthrope_fade.wav', frequency, combo)