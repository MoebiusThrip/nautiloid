# import reload
from importlib import reload

# import general tools
import os, json

# import datetime
from datetime import datetime, timedelta
from time import time

# import math
from math import exp, sqrt, sin, cos, pi, log2
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
        self.audible = (0, 0)
        self._tune()

        # current song parameters
        self.name = ''
        self.rate = 0
        self.size = 0
        self.length = 0

        # store fourier analysis
        self.fourier = []

        # analytical parameters
        self.tolerance = 0.1

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
        size = int(len(sequence) / number)

        # make pieces
        pieces = [sequence[size * index: size + size * index] for index in range(number)]

        return pieces

    def _draw(self, lines, texts, destination, annotations=None):
        """Draw a group of lines.

        Arguments:
            lines: list of tuples
            texts: list of str
            destination: str, filepath
            annotations=None

        Returns:
            None
        """

        # set default annotations
        annotations = annotations or []

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

        # add annotations
        for horizontal, vertical, text in annotations:

            # add annotations
            axes[0].annotation(text, xy=(horizontal, vertical))
            axes[1].annotation(text, xy=(horizontal, vertical))
            axes[2].annotation(text, xy=(horizontal, vertical))

        # add legend
        figure.legend(loc='upper right')

        # save the figure
        pyplot.savefig(destination)

        return None

    def _dump(self, contents, destination):
        """Dump a dictionary into a json file.

        Arguments:
            contents: dict
            destination: str, destination file path

        Returns:
            None
        """

        # dump file
        with open(destination, 'w') as pointer:

            # dump contents
            print('dumping into {}...'.format(destination))
            json.dump(contents, pointer)

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

    def _lay(self):
        """Lay a grid based on the song.

        Arguments:
            None

        Returns:
            numpy array
        """

        # timestamp
        self._stamp('laying grid...', initial=True)

        # get maximum length
        length = max([len(sixteenth) for measure in list(self) for sixteenth in measure])

        # make grid
        grid = [[index * indexii for indexii in range(length)] for index in range(length)]
        grid = numpy.array(grid)

        # timestamp
        self._stamp('layed.')

        return grid

    def _load(self, path):
        """Load a json file.

        Arguments:
            path: str, file path

        Returns:
            dict
        """

        # try to
        try:

            # open json file
            with open(path, 'r') as pointer:

                # get contents
                print('loading {}...'.format(path))
                contents = json.load(pointer)

        # unless the file does not exit
        except FileNotFoundError:

            # in which case return empty json
            print('creating {}...'.format(path))
            contents = {}

        return contents

    def _normalize(self, data):
        """Normalize a list of data to its z-scores.

        Arguments:
            data: list of floats

        Returns:
            list of floats
        """

        # substract mean and divide by standard deviation
        mean = numpy.average(data)
        deviation = numpy.std(data)

        # make data
        normalization = numpy.array([(datum - mean) / deviation for datum in data])

        return normalization

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

    def _resonate(self, frequency, amplitude):
        """Find the closest matches to each frequency.

        Arguments:
            note: float, frequency of note in hertz

        Returns:
            list of (str, float) tuples
        """

        # make a score from each note
        scores = []
        for name, note in self.forks.items():

            # create score
            score = (round(frequency, 2), round(amplitude, 2), round(1200 * log2(frequency / note), 2), name)
            scores.append(score)

        # score each entry in the tuning forks and sort based on closeness
        scores.sort(key=lambda item: (item[2] - 1) ** 2)

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

    def _transform(self, snippet, grid):
        """Calculate the fourier transform.

        Arguments:
            snippet: list of ints
            grid: 2-D number area of index

        Returns:
            list of floats
        """

        # get the total number of frames
        number = len(snippet)

        # subset grid in case length differs
        grid = grid[:number, :number]

        # multiple by 2 pi / N
        cycles = grid * (2 * pi / number)

        # take the cosine
        cosines = numpy.cos(cycles)

        # get amplitudes
        amplitudes = numpy.tile(snippet, (number, 1))

        # multiple to get terms
        terms = cosines * amplitudes

        # sum all terrms
        fourier = terms.sum(axis=1)

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

        # get upper and lower frequency ranges
        self.audible = (min(forks.values()), max(forks.values()))

        return None

    def forage(self, fourier, spectrum, faintness=1.0):
        """Find the peaks in the fourier spectrum.

        Arguments:
            fourier: list of floats
            spectrum: list of floats
            faintness: float, the cutoff amplitude for a peak

        Returns:
            None
        """

        # search for peaks
        peaking = lambda series, index: series[index] > series[index - 1] and series[index] > series[index + 1]
        peaks = [index + 1 for index, _ in enumerate(fourier[1:-1]) if peaking(fourier, index + 1)]

        # find all peaks in audible range and of high enough amplitude
        peaks = [peak for peak in peaks if fourier[peak] > faintness]
        peaks = [peak for peak in peaks if self.audible[0] < spectrum[peak] < self.audible[1]]

        # score each peak and print
        scores = [self._resonate(spectrum[peak], fourier[peak]) for peak in peaks]
        [print('{} Htz, {} std ({} cents): {}'.format(*score[0])) for score in scores]

        return None

    def ink(self, measure, sixteenth, display=True):
        """Draw the spectrum for the particular sixteenth.

        Arguments:
            measure: int, measure index
            sixteenth: int, sixteenth index
            display: boolean, create plot?

        Returns:
            None
        """

        # get snippet and fourier results
        snippet = self[measure][sixteenth]
        fourier = self.fourier[measure][sixteenth]

        # convert fourier spectrum to frequencies
        spectrum = numpy.array([self.rate / (index or 1) for index, _ in enumerate(fourier)])

        # normalize the fourier
        fourier = [amplitude ** 2 for amplitude in fourier]
        fourier = self._normalize(fourier)

        # find the notes
        self.forage(fourier, spectrum)

        # make plot
        if display:

            # normalize the snippet and reverse
            snippet = [float(entry) for entry in self._normalize(snippet).data]
            snippet.reverse()
            snippet = numpy.array(snippet)

            # get the list of indices for audible range of hearing
            audibles = [index for index, frequency in enumerate(spectrum) if self.audible[0] < frequency < self.audible[1]]

            # subset the fourier and spectrum
            fourier = fourier[audibles]
            spectrum = spectrum[audibles]

            # interpolate spectrum to graph snippet
            final = spectrum[0]
            span = (spectrum[0] - spectrum[-1]) / len(snippet)
            interpolation = [final - index * span for index, amplitude in enumerate(snippet)]

            # create the fourier line
            line = (spectrum, fourier, 'b-', 'fourier')
            lines = [line]

            # add plot?
            lineii = (interpolation, snippet, 'g-', 'snippet')
            lines.append(lineii)

            # add labels
            title = 'fourier transform of {}'.format(self.name)
            independent = 'frequency'
            dependent = 'amplitude'
            texts = [title, independent, dependent]

            # make destination
            destination = 'plots/{}_{}:{}.png'.format(self.name, measure, sixteenth)

            # draw it
            self._draw(lines, texts, destination)

        return None

    def listen(self, name, size=44):
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

        # start clock
        self._stamp('listening to {}...'.format(song), initial=True)

        # open the wave file
        rate, data = wavfile.read(song)

        # set song parameters
        self.name = name
        self.rate = rate
        self.size = size
        self.length = len(data)

        # combine both channels
        data = [sum(datum) for datum in data]

        # chop into measures and then into sixteenths
        measures = self._dice(data, size)
        sixteenths = [self._dice(measure, 16) for measure in measures]

        # populate
        [self.append(measure) for measure in sixteenths]

        # timestamp
        self._stamp('listened.')

        # print stats
        print('length: {} frames, {} seconds'.format(self.length, round(self.length / self.rate), 2))
        print('sampling rate: {} frames / s'.format(self.rate))

        return None

    def memorize(self):
        """Store the fourier analysis results

        Arguments:
            None

        Returns:
            None
        """

        # convert current fourier into floats
        memory = [[[float(amplitude) for amplitude in sixteenth] for sixteenth in measure] for measure in self.fourier]

        # store in fouriers
        destination = 'fouriers/{}_fourier.json'.format(self.name)
        self._dump(memory, destination)

        return None

    def recognize(self):
        """Load in previous fourier analysis for this song.

        Arguments:
            None

        Returns:
            None
        """

        # construct file path
        path = 'fouriers/{}_fourier.json'.format(self.name)
        memory = self._load(path)

        # convert to numpy arrays
        fourier = [[numpy.array(sixteenth) for sixteenth in measure] for measure in memory]

        # set attribute
        self.fourier = fourier

        return None

    def undulate(self):
        """Get the frequency spectrum for a snippet of a song.

        Arguments:
            None

        Returns:
            None
        """

        # compute index grid
        grid = self._lay()

        # begin fourier analysis
        fourier = []

        # for each measure
        for index, measure in enumerate(self):

            # timestamp
            self._stamp('transforming measure {} of {}...'.format(index, len(self)), initial=True)

            # analyze each sixteenth note
            analyses = []
            for indexii, sixteenth in enumerate(measure):

                # perform fourier transform on sixteenth note
                print('semiquaver {} of 16...'.format(indexii))
                analysis = self._transform(sixteenth, grid)
                analyses.append(analysis)

            # add to fourier
            fourier.append(analyses)

            # timestamp
            self._stamp('transformed.')

        # add to record
        self.fourier = fourier

        return None

