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

        # tune the frequencies
        self.forks = {}
        self._tune()

        # default song parameters
        self.measures = 40
        self.rate = 0
        self.length = 0

        # store fourier analysis
        self.fourier = []

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

    def _dice(self, sequence, number):
        """Dice a sequence into a number of chunks.

        Arguments:
            sequence: list to floats
            number: int, number pieces

        Returns:
            list of list of floats
        """

        # determine chunk size
        size = intt(sequence / number)

        # make pieces
        pieces = [sequence[size * index: size + size * index] for index in range(number)]

        return pieces

    def _draw(self, lines, texts, destination):
        """Draw a group of lines.

        Arguments:
            lines: list of tuples
            texts: list of str
            destination: str, filepath

        Returns:
            None
        """

        # begin plot
        pyplot.clf()
        figure = pyplot.figure()

        # create a grid for combining subplots
        grid = figure.add_gridspec(ncols=1, nrows=3, width_ratios=[1], height_ratios=[2, 10, 2])

        # defie the axes
        axes = [figure.add_subplot(grid[0, :])]
        axes.append(figure.add_subplot(grid[1, :]))
        axes.append(figure.add_subplot(grid[2, :]))

        # adjust margins
        figure.subplots_adjust(hspace=0.0, wspace=0.5)

        # determine all entries
        data = [float(datum) for line in lines for datum in line[1]]

        # define quantile cutoffs for head and tail
        quantile = 2
        maximum = max(data)
        head = numpy.percentile(data, 100 - quantile)
        tail = numpy.percentile(data, quantile)
        minimum = min(data)

        # plot all lines
        for abscissa, ordinate, style, label in lines:

            # if there is a label
            if label:

                # plot with a label
                axes[0].plot(abscissa, ordinate, style)
                axes[1].plot(abscissa, ordinate, style, label=label)
                axes[2].plot(abscissa, ordinate, style)

            # otherwise
            else:

                # plot without a label
                axes[0].plot(abscissa, ordinate, style)
                axes[1].plot(abscissa, ordinate, style)
                axes[2].plot(abscissa, ordinate, style)

        # set axis limits
        axes[0].set_ylim(head, maximum)
        axes[1].set_ylim(tail, head)
        axes[2].set_ylim(minimum, tail)

        # parse text, padding with blanks
        texts = texts + ['', '', '']
        title, independent, dependent = texts[:3]

        # add labels
        axes[0].set_title(title)
        axes[1].set_ylabel(dependent)
        axes[2].set_xlabel(independent)

        # remove xtick labels in head and main
        axes[0].set_xticklabels([" "] * len(axes[1].get_xticks()))
        axes[1].set_xticklabels([" "] * len(axes[1].get_xticks()))

        # add legend
        figure.legend(loc='upper right')

        # save the figure
        pyplot.savefig(destination)

        return None

    def _forget(self):
        """Depopulate current song.

        Arguments:
            None

        Returns:
            None
        """

        # as long as there is memory
        while len(self) > 0:

            # forget
            self.pop()

        return None

    def _listen(self, name):
        """Get a wave file by listening to a song.

        Arguments:
            name: partial song title

        Returns:
            None
        """

        # forget current song
        self._forget()

        # get all paths
        paths = self._see(self.directory)

        # get all paths with the name, assuming the first is correct
        song = [path for path in paths if name.lower() in path.lower()][0]

        # open the wave file
        rate, data = wavfile.read(song)

        # set song parametesr
        self.rate = rate
        self.length = len(data)

        # combine both channels
        data = [sum(datum) for datum in data]

        # chop into measures and then into sixteenths
        measures = self._dice(data, self.measures)
        sixteenths = [self._dice(measure, 16) for measure in measures]

        # populate
        [self.append(measure) for measure in sixteenths]

        return None

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

    def _resonate(self, note):
        """Find the closest matches to each frequency.

        Arguments:
            note: float, frequency of note in hertz

        Returns:
            list of (str, float) tuples
        """

        # score each entry in the tuning forks and sort based on closeness
        scores = [(name, round(note / frequency, 2)) for name, frequency in self.forks.items()]
        scores.sort(key=lambda item: (item[1] - 1) ** 2)

        return scores

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

    def _transcribe(self, path):
        """Transcribe the text file at the path.

        Arguments:
            path: str, file path

        Returns:
            list of str
        """

        # read in file pointer
        with open(path, 'r') as pointer:

            # read in text and eliminate endlines
            lines = pointer.readlines()
            lines = [line.replace('\n', '') for line in lines]

        return lines

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

    def _tune(self):
        """Establish frequencies.

        Arguments:
            None

        Returns:
            None

        Populates:
            self.forks
        """

        # transcribe the frequency table, skipping the header
        table = self._transcribe('theory/frequencies.txt')[1:]

        # create tuning forks
        forks = {note.strip(): float(frequency) for note, frequency, _ in [entry.split('\t') for entry in table]}
        self.forks.update(forks)

        return None

    def undulate(self, name, size, start=0, hertz=4000):
        """Get the frequency spectrum for a snippet of a song.

        Arguments:
            name: (partial) name of song
            size: number of frames
            start: starting position
            hertz: highest frequency in hertz to plot

        Returns:
            None
        """

        # listen to the song
        self._listen(name)
        print('length: {} frames, {} seconds'.format(self.length, round(self.length / self.rate), 2))
        print('sampling rate: {} frames / s'.format(self.rate))

        # begin fourier analysis
        fourier = []

        # for each measure
        for measure in list(self):

            # analyze each sixteenth note
            analysis = [self._transform(sixteenth) for sixteenth in measure]
            fourier.append(analysis)

        # add to record
        self.fourier = fourier





        # get the first channel snippet
        snippet = song[start:start + size, 0]


        # make spectrum from frequency
        spectrum = [frequency / (index + 1) for index, _ in enumerate(snippet)]

        # get index closet to set hertz level
        musical = [index for index, wave in enumerate(spectrum) if wave < hertz][0]

        # create the line
        line = (spectrum[musical:], fourier[musical:], 'b-', 'fourier')
        lines = [line]

        # add plot?
        lineii = (spectrum[musical:], snippet[musical:], 'g-', 'snippet')
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

        # search for peaks
        peaking = lambda series, index: series[index] > series[index - 1] and series[index] > series[index + 1]
        peaks = [index + 1 for index, _ in enumerate(fourier[1:-1])if peaking(fourier, index + 1)]

        # print the frequencies
        notes = [spectrum[peak] for peak in peaks if spectrum[peak] > 500]
        scores = [self._resonate(note) for note in notes]
        [print('{} Htz: {} ({})'.format(note, *score[0])) for note, score in zip(notes, scores)]

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