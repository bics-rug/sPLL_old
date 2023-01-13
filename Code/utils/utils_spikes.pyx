import matplotlib.pyplot as plt
import numpy as np
from brian2 import *

from utils_plot import *
from utils_spikes import *
from utils_misc import *
from utils_info import *
# from Pacini_Encoding import enconding_multiple_frequencies
from brian2 import *

class dummy_spikestructure:
    '''
    Structure for storing spikes, similar to Brian2
    '''

    def __init__(self, t, i, sinusoid=[],verbose = False):
        '''
        Initialization fuction of the structure. A time and index of spikes is required
        :param t: the times of each single spikes, in an array. The array can be adimensional or seconds
        :param i: the index of the neuron that is spiking, in an array
        :param sinusoid: (optional) a sinusoid can be provided to generate spikes from it (phase locked with the peak)
        '''
        self.i = np.array(i)
        self.t = self.set_dimension(t, second)
        self.verbose = verbose
        if self.ismultipleidx():
            self.dummyspike_collection = []
            self.divide_indexes(self.t,self.i,sinusoid)
            self.avg_isi_value = []
            self.it = [self.t, self.i]
            self.count = np.array([])
            self.isi_short = []
            self.isi_long = []
            self.info = {}
        else:
            # self.t = self.set_dimension(t, second)
            # self.fix_minimum_timestep()
            # self.i = np.array(i)
            self.fix_minimum_timestep()
            self.it = [self.t, self.i]
            self.tdiscrete = np.array([])
            self.count = np.array([])
            self.fft = []
            self.fftfreqs = []
            self.fs = None
            self.sinusoid = sinusoid
            self.count_in_interval = []
            self.upper_time_intervals = []
            self.isi = {'value': [],'time': []}
            self.instfreq = {'value': [],'time': []}
            self.avg_isi_value = None
            self.isi_short = None
            self.isi_long = None

    def ismultipleidx(self):
        return len(np.unique(self.i)) > 1
    def set_dimension(self, value, dimension):
        '''
        Set the dimension of a variable, if it doesn't have it
        :param value: the list/value/array that require dimension
        :param dimension: the dimension we want to give
        :return: the value with the dimension
        '''
        if get_dimensions(value) != dimension:
            if str(get_dimensions(value)) == 's':
                value_to_return = value / dimension
                value_to_return = value_to_return * dimension
            else:
                value_to_return = value * dimension
        else:
            value_to_return = value
        return value_to_return
    def save_data(self, path):
        np.savetxt(path, self.it)
    def bin_in_time(self, fs):
        '''
        Function that bin the spikes into discrete time
        :param fs: The sampling frequency that discretize it
        :return: The binned spikes
        '''
        if self.ismultipleidx():
            for dix,dummy_structure in enumerate(self.dummyspike_collection):
                update_progress(dix/len(self.dummyspike_collection), 'bin in time in progress...')
                dummy_structure.bin_in_time(fs = fs)
        else:
            assert fs != None, 'When calling the bin_in_time() or fft related stuff remember to provide fs'
            self.fs = self.set_dimension(fs, hertz)
            low_time_boundary = 0 * second
            if len(self.t) > 0:
                steps = int(np.round(fs * second / self.t.max()))
            else:
                steps = 0
            for step in range(steps):
                # update_progress((step+1)/steps, 'bin in time in progress...')
                high_time_boundary = (step + 1) * 1 / self.fs
                self.count = np.append(self.count,
                                       len(self.t[(self.t >= low_time_boundary) & (self.t < high_time_boundary)]))
                self.tdiscrete = np.append(self.tdiscrete, low_time_boundary)
                low_time_boundary = high_time_boundary

    def count_in_intervals(self, upper_time_intervals=[]):
        '''
        A function that counts how many spikes are present in given intervals.
        If no upper_time_intervals are given, the function calculate the total spike count
        :param upper_time_intervals: A list of time values that identify the boundaries of different time intervals
        :return: the spike count in the intervals. If no upper_time_inverval is provided, it returns a int with the total spike count
        '''
        if self.ismultipleidx():
            for dummy_structure in self.dummyspike_collection:
                dummy_structure.count_in_intervals(upper_time_intervals = upper_time_intervals)
                self.count = np.append(self.count,dummy_structure.count_in_interval)

        else:
            # count_in_interval = np.array([])
            if len(upper_time_intervals) == 0:
                upper_time_intervals = [self.t.max()]
            low_time_boundary = 0 * second
            for high_time_boundary in upper_time_intervals:
                high_time_boundary = self.set_dimension(high_time_boundary, second)
                self.count_in_interval = np.append(self.count_in_interval,
                                              len(self.t[(self.t >= low_time_boundary) & (
                                                          self.t < high_time_boundary)]))
                low_time_boundary = high_time_boundary
                self.upper_time_intervals = upper_time_intervals
    def plot_in_intervals(self,upper_time_intervals = [], offset=0, gain=1, norm=False, label=None,
                               plot_where=None,
                               return_s=False, color=None):
        if self.ismultipleidx():
            idx = 0
            for dummy_structure in self.dummyspike_collection:
                dummy_structure.plot_in_intervals(upper_time_intervals = upper_time_intervals, offset=offset+idx,
                                                  gain=gain, norm=norm, label=label, plot_where=plot_where,
                               return_s=return_s, color=color)
                idx += 1
        else:
            if len(self.count_in_interval) == 0:
                self.count_in_intervals(upper_time_intervals = upper_time_intervals)
            plot(upper_time_intervals,gain * self.count_in_interval / (
                             self.count_in_interval.max() * int(norm) + 1) + offset, color = color)
    def compute_fft(self, fs=None):
        '''
        A function that compute the Fast Fourier Transform of the spike train. If no binning was already done it
        automatically perform it, in this case fs should be provided.
        :param fs: sampling frequency for bin_in_time
        :return: the fast fourier transform
        '''

        if self.ismultipleidx():
            for dix, dummy_structure in enumerate(self.dummyspike_collection):
                update_progress(dix / len(self.dummyspike_collection) + 1, 'compute_fft in progress...')
                dummy_structure.compute_fft(fs = fs)
        else:
            if len(self.count) == 0:
                self.bin_in_time(fs)
            self.fft = fft(self.count)
            self.fftfreqs = fftfreq(len(self.count), d=1 / self.fs)
    def calculate_instfreq(self, average = 1,average_percentage = None):
        if self.ismultipleidx():
            for dummy_structure in self.dummyspike_collection:
                dummy_structure.calculate_instfreq(average = average,average_percentage = average_percentage)

        else:
            if len(self.isi['value']) == 0:
                self.isi_calculator(average = average,average_percentage = average_percentage)
            self.instfreq['value'] = 1/self.isi['value']
            self.instfreq['time'] = self.isi['time']
    def plot_instfreq(self,average = 1, offset=0, gain=1, norm=False, label=None,
                               plot_where=None, neuron_list = [], x_offset = 0, color = None, alpha = None, average_percentage = None):
        if self.ismultipleidx():
            if plot_where == None:
                _, plot_where = plt.subplots(nrows=1, ncols=1)
            idx = 0
            if len(neuron_list) == 0:
                neuron_list = [i for i in range(len(self.dummyspike_collection))]
            for d_ix, dummy_structure in enumerate(self.dummyspike_collection):
                if d_ix in neuron_list:
                    dummy_structure.plot_instfreq(average = average, offset = offset+idx, gain = gain, norm = norm, label = label,
                                         plot_where = plot_where, color = color, average_percentage = average_percentage)
                    # idx += 1
        else:
            if len(self.instfreq['value']) == 0:
                self.calculate_instfreq(average = average)
            self.plot_function(self.instfreq['time'], self.instfreq['value'], norm=norm, gain=gain, offset=offset,
                               label=label, plot_where=plot_where, x_offset = x_offset, color = color, alpha = alpha)
    def plot_function(self, x, y, norm = False, offset=0, gain=1, min_x=None, max_x=None, label=None,
                       plot_where=None, color=None, x_offset = 0,alpha = None):
        if len(x) > 0:
            if plot_where == None:
                _,plot_where = plt.subplots(nrows=1, ncols=1)
            if min_x == None:
                try:
                    min_x = x.min()
                except ValueError:
                    print('we')
            if max_x == None:
                max_x = x.max()
            plot_where.plot(x[(x >= min_x) & (x < max_x)] + x_offset,
                 gain * y[(x >= min_x) & (x < max_x)] / (
                         y.max() * int(norm) + 1) + offset,
                 label=label, marker='.', color=color, alpha = alpha)
    def fix_minimum_timestep(self,timestep_fix = 10*us):
        # print('t_before: ' + str(len(self.t)))
        # print('i_before: ' + str(len(self.i)))
        if (self.ismultipleidx() == False) & (type(self.t/second) == numpy.ndarray):
                idx = np.argsort(self.t)
                # print(idx)
                try:
                    self.t = self.t[idx]
                except IndexError:
                    print(self.t)
                    if np.isnan(self.t) == True:
                        pass
                # print(self.t[0])
                try:
                    self.i = self.i[idx]
                except IndexError:
                    print('halo')
                timediff = np.diff(self.t)
                timediff_toosmall = np.where(timediff <= timestep_fix)
                while len(timediff_toosmall[0]) > 0:
                    for timediff_toosmall_single in timediff_toosmall:
                        self.t = self.set_dimension(np.delete(self.t,timediff_toosmall_single),second)
                        self.i = np.delete(self.i,timediff_toosmall_single)
                    timediff = np.diff(self.t)
                    try:
                        timediff_toosmall = np.where(timediff<=timestep_fix)
                    except DimensionMismatchError:
                        print('wee')
        # print('t_after: ' + str(len(self.t)))
        # print('i_after: ' + str(len(self.i)))
    def plot_fft(self, norm=False, offset=0, gain=1, min_x=None, max_x=None, label=None, fs=None,
                 return_s=False,
                 plot_where=None, color=None, show = True):
        '''
        This function plots the fft computed in the function compute_fft. If no fft is ready the function calls compute_fft
        :param norm: (boolean) Flag for defining if the values should be normalized for the maximum value (used for multiple plots in 1 plot)
        :param offset: (float) Value for defining where the 0 should be in the plot (used for multiple plots in 1 plot)
        :param gain: (float) Value for defining where the maximum value should be in the plot (used for multiple plots in 1 plot)
        :param min_x: (float) Value for defining the x_axis minimum
        :param max_x: (float) Value for defining the x_axis maximum
        :param label: (str) String that can be appended to the plot for the legend (used for multiple plots in 1 plot)
        :param fs: (float) Frequency of sampling for bin_in_time (needed only if binning doesn't exist yet)
        :param return_s: (boolean) Flag that defines if the resulting dummy_spikestructure should be returned or not
        :param plot_where: (axis object) Define on which figure the plot should go (useful when having multiple plots on different windows)
        :param color: (color) Define the color of the plot
        :return: If return_s is True the dummy_spikestructure is returned
        '''
        if self.ismultipleidx():
            idx = 0
            for dix, dummy_structure in enumerate(self.dummyspike_collection):
                update_progress((dix+1) / len(self.dummyspike_collection), 'fft_plot in progress')
                dummy_structure.plot_fft(norm=norm, offset=offset + idx, gain=gain, min_x=min_x, max_x=max_x,
                                         label=label, fs=fs,
                                         return_s=return_s,
                                         plot_where=plot_where, color=color)
                idx += 1
        else:
            if len(self.fft) == 0:
                self.compute_fft(fs)
            if min_x == None:
                min_x = self.fftfreqs.min()
            if max_x == None:
                max_x = self.fftfreqs.max()
            if plot_where == None:
                plot(self.fftfreqs[(self.fftfreqs > min_x) & (self.fftfreqs < max_x)],
                     np.abs(gain * self.fft[(self.fftfreqs > min_x) & (self.fftfreqs < max_x)] / (
                             self.fft.max() * int(norm) + 1)) + offset,
                     label=label, marker='.', color=color)
                # fill_between(self.fftfreqs[(self.fftfreqs > min_x) & (self.fftfreqs < max_x)],
                #      gain * self.fft[(self.fftfreqs > min_x) & (self.fftfreqs < max_x)] / (
                #              self.fft.max() * int(norm) + 1) + offset, color = color, alpha = 0.8)
            else:
                plot_where.plot(self.fftfreqs[(self.fftfreqs > min_x) & (self.fftfreqs < max_x)],
                                np.abs(gain * self.fft[(self.fftfreqs > min_x) & (self.fftfreqs < max_x)] / (
                                        self.fft.max() * int(norm) + 1)) + offset,
                                label=label, marker='.', color=color)
                # plot_where.fill_between(self.fftfreqs[(self.fftfreqs > min_x) & (self.fftfreqs < max_x)],
                #      gain * self.fft[(self.fftfreqs > min_x) & (self.fftfreqs < max_x)] / (
                #              self.fft.max() * int(norm) + 1) + offset, color = color, alpha = 0.8)

            if return_s == True:
                return self
            if show == True:
                plt.show()


    def plot_count_vs_interval(self, upper_time_intervals=[], offset=0, gain=1, norm=False, label=None,
                               plot_where=None,
                               return_s=False, color=None):
        '''
        This function plots the spike count over an interval created with count_in_intervals.
        If no fft is ready the function calls count_in_intervals
        :param (list) upper_time_intervals: A list of upper limit for intervals
        :param (boolean) norm: Flag for defining if the values should be normalized for the maximum value (used for multiple plots in 1 plot)
        :param (float) offset: Value for defining where the 0 should be in the plot (used for multiple plots in 1 plot)
        :param (float) gain: Value for defining where the maximum value should be in the plot (used for multiple plots in 1 plot)
        :param (float) min_x: Value for defining the x_axis minimum
        :param (float) max_x: Value for defining the x_axis maximum
        :param (str) label: String that can be appended to the plot for the legend (used for multiple plots in 1 plot)
        :param (float) fs: Frequency of sampling for bin_in_time (needed only if binning doesn't exist yet)
        :param (boolean) return_s: Flag that defines if the resulting dummy_spikestructure should be returned or not
        :param (axis object) plot_where: Define on which figure the plot should go (useful when having multiple plots on different windows)
        :param (color) color: Define the color of the plot
        :return: If return_s is True the dummy_spikestructure is returned
        '''
        if len(upper_time_intervals) != 0:
            value = self.count_in_intervals(upper_time_intervals)
            if plot_where == None:
                plot(gain * value / (value.max() * int(norm) + 1) + offset, label=label)
            else:
                plot_where.plot(gain * value / (value.max() * int(norm) + 1) + offset, label=label)
        else:
            # print('No upper time intervals given, assuming one big interval with the entire nmulation time')
            return self.count_in_intervals(upper_time_intervals)
    def find_long_isi(self,time_threshold = 10*ms):
        if self.ismultipleidx():
            self.isi_long = []
            for dummy_structure in self.dummyspike_collection:
                dummy_structure.find_long_isi(time_threshold=time_threshold)
                self.isi_long.append(dummy_structure.isi_long)
        else:
            if len(self.isi['value']) == 0:
                self.isi_calculator()
            print(self.isi['value'])
            self.isi_long = self.isi['value'][self.isi['value'] > time_threshold]

    def find_short_isi(self,time_threshold = 1*ms):
        if self.ismultipleidx():
            self.isi_short = []
            for dummy_structure in self.dummyspike_collection:
                dummy_structure.find_short_isi(time_threshold = time_threshold)
                self.isi_short.append(dummy_structure.isi_short)
        else:
            if len(self.isi['value']) == 0:
                self.isi_calculator()
            self.isi_short = self.isi['value'][self.isi['value']*second < time_threshold]


    def isi_calculator(self, average = 1,average_percentage = None):
        '''
        This function calculates interspike intervals
        :param average (int) the averaging factor of the sliding mean
        :return:
        '''
        if self.ismultipleidx():
            for dummy_structure in self.dummyspike_collection:
                dummy_structure.isi_calculator(average = average,average_percentage = average_percentage)

        else:
            # if average_percentage != None:
            #     average = int(np.floor((average/100)*len(self.t)))
            if len(self.t) > average:
                self.isi['value'] = np.convolve(np.diff(self.t), np.ones(average)/average, mode='valid')
            else:
                self.isi['value'] = np.diff(self.t)
            if (average > 1) & (len(self.t) > average):
                self.isi['time'] = self.t[1:-int(average-1)]
            elif len(self.t) > 1:
                self.isi['time'] = self.t[1:]
            else:
                self.isi['time'] = np.NaN
    def avg_isi(self):
        self.avg_isi_value  = np.array([])
        if self.ismultipleidx():
            for dummy_structure in self.dummyspike_collection:
                dummy_structure.avg_isi()
                self.avg_isi_value = np.append(self.avg_isi_value,dummy_structure.avg_isi_value)
        else:
            if len(self.isi['value']) == 0:
                self.isi_calculator()
            self.avg_isi_value = np.average(self.isi['value'])
    def plot_isilong_vs_intrinsicfreqandinputfrequency(self,neurons_n, frequencies, s_osc_intrinsic, norm = False, cmap = None, min_max = None, img_title = None):
        if cmap == None:
            cmap = 'Purples'
        freq = 0
        neuron = 0
        my_matrix = np.zeros([neurons_n,len(frequencies)])
        if len(self.isi_short) == 0:
            self.find_long_isi()
        for dummy_structure_ix in range(len(self.dummyspike_collection)):
            my_matrix[neuron, freq] = len(self.isi_long[dummy_structure_ix])
            if neuron == neurons_n - 1:
                neuron = 0
                freq += 1
            else:
                neuron += 1
        if min_max != None:
            if min_max == 'min':
                min_max_list = my_matrix.min(axis=0)
            elif min_max == 'max':
                min_max_list = my_matrix.max(axis=0)
            min_max_where = []
            for idx, min_max_here in enumerate(min_max_list):
                min_max_where.append(np.where(my_matrix[:, idx] == min_max_here)[0][0])

            plot(min_max_where, '.', color='k', linewidth=2)
        if norm == False:
            norm = 1
        else:
            norm = my_matrix.max(axis=0)
        imshow(my_matrix / norm, cmap= cmap)
        # plot(min/norm)
        xlabel('Input Frequency (Hz)')
        ylabel('Intrinsic Frequency (Hz)')
        title(img_title)
        ticks = Reduce_ticks_labels(np.round(np.array(frequencies)), ticks_steps=10)
        xticks([i for i in range(len(frequencies))], ticks)
        ticks = Reduce_ticks_labels(np.round(1 / s_osc_intrinsic.avg_isi_value), ticks_steps=5)
        yticks([i for i in range(neurons_n)], ticks)
        colorbar()
    def plot_isishort_vs_intrinsicfreqandinputfrequency(self,neurons_n, frequencies, s_osc_intrinsic, norm = False, cmap = None, min_max = None, img_title = None):
        if cmap == None:
            cmap = 'Purples'
        freq = 0
        neuron = 0
        my_matrix = np.zeros([neurons_n,len(frequencies)])
        if len(self.isi_short) == 0:
            self.find_short_isi()
        for dummy_structure_ix in range(len(self.dummyspike_collection)):
            my_matrix[neuron, freq] = len(self.isi_short[dummy_structure_ix])
            if neuron == neurons_n - 1:
                neuron = 0
                freq += 1
            else:
                neuron += 1
        if min_max != None:
            if min_max == 'min':
                min_max_list = my_matrix.min(axis=0)
            elif min_max == 'max':
                min_max_list = my_matrix.max(axis=0)
            min_max_where = []
            for idx, min_max_here in enumerate(min_max_list):
                min_max_where.append(np.where(my_matrix[:, idx] == min_max_here)[0][0])

            plot(min_max_where, '.', color='k', linewidth=2)
        if norm == False:
            norm = 1
        else:
            norm = my_matrix.max(axis=0)
        imshow(my_matrix / norm, cmap= cmap)
        # plot(min/norm)
        xlabel('Input Frequency (Hz)')
        ylabel('Intrinsic Frequency (Hz)')
        stimuli = []
        for i in range(len(frequencies)):
            stimuli += [i for j in range(neurons_n)]
        info = self.calculate_info(stimuli = stimuli,code = 'ISI_short',winner = min_max)
        self.info[img_title] = info
        title(img_title+ '. Info (bits): ' + str(info))
        ticks = Reduce_ticks_labels(np.round(np.array(frequencies)), ticks_steps=10)
        xticks([i for i in range(len(frequencies))], ticks)
        ticks = Reduce_ticks_labels(np.round(1 / s_osc_intrinsic.avg_isi_value), ticks_steps=5)
        yticks([i for i in range(neurons_n)], ticks)
        colorbar()
    def plot_isilong_vs_intrinsicfreqandinputfrequency(self,neurons_n, frequencies, s_osc_intrinsic, norm = False, cmap = None, min_max = None, img_title = None):
        if cmap == None:
            cmap = 'Purples'
        freq = 0
        neuron = 0
        my_matrix = np.zeros([neurons_n,len(frequencies)])
        if len(self.isi_long) == 0:
            self.find_long_isi()
        for dummy_structure_ix in range(len(self.dummyspike_collection)):
            my_matrix[neuron, freq] = len(self.isi_long[dummy_structure_ix])
            if neuron == neurons_n - 1:
                neuron = 0
                freq += 1
            else:
                neuron += 1
        if min_max != None:
            if min_max == 'min':
                min_max_list = my_matrix.min(axis=0)
            elif min_max == 'max':
                min_max_list = my_matrix.max(axis=0)
            min_max_where = []
            for idx, min_max_here in enumerate(min_max_list):
                min_max_where.append(np.where(my_matrix[:, idx] == min_max_here)[0][0])

            plot(min_max_where, '.', color='k', linewidth=2)
        if norm == False:
            norm = 1
        else:
            norm = my_matrix.max(axis=0)
        imshow(my_matrix / norm, cmap= cmap)
        # plot(min/norm)
        xlabel('Input Frequency (Hz)')
        ylabel('Intrinsic Frequency (Hz)')
        stimuli = []
        for i in range(len(frequencies)):
            stimuli += [i for j in range(neurons_n)]
        info = self.calculate_info(stimuli = stimuli,code = 'ISI_long',winner = min_max)
        self.info[img_title] = info
        title(img_title + '. Info (bits): ' + str(info))
        # title(img_title)
        ticks = Reduce_ticks_labels(np.round(np.array(frequencies)), ticks_steps=10)
        xticks([i for i in range(len(frequencies))], ticks)
        ticks = Reduce_ticks_labels(np.round(1 / s_osc_intrinsic.avg_isi_value), ticks_steps=5)
        yticks([i for i in range(neurons_n)], ticks)
        colorbar()
    def plot_isi_vs_intrinsicfreqandinputfrequency(self, neurons_n, frequencies, s_osc_intrinsic, norm = False, cmap = None, min_max = None, img_title = None, show = True):
        if cmap == None:
            cmap = 'Purples'
        freq = 0
        neuron = 0
        my_avg = np.zeros([neurons_n,len(frequencies)])
        if len(self.avg_isi_value) == 0:
            self.avg_isi()
        for dummy_structure_ix in range(len(self.dummyspike_collection)):
            my_avg[neuron, freq] = self.avg_isi_value[dummy_structure_ix]
            if neuron == neurons_n - 1:
                neuron = 0
                freq += 1
            else:
                neuron += 1
        if min_max != None:
            if min_max == 'min':
                min_max_list = my_avg.min(axis=0)
            elif min_max == 'max':
                min_max_list = my_avg.max(axis=0)
            min_max_where = []
            for idx, min_max_here in enumerate(min_max_list):
                min_max_where.append(np.where(my_avg[:, idx] == min_max_here)[0][0])

            plot(min_max_where, '.', color='k', linewidth=2)
        if norm == False:
            norm = 1
        else:
            norm = my_avg.max(axis=0)
        imshow(my_avg / norm, cmap= cmap,aspect = 'auto')
        # plot(min/norm)
        xlabel('Input Frequency (Hz)')
        ylabel('Intrinsic Frequency (Hz)')
        stimuli = []
        for i in range(len(frequencies)):
            stimuli += [i for j in range(neurons_n)]
        # info = self.calculate_info(stimuli = stimuli,code = 'ISI_mean',winner = min_max)
        # self.info[img_title] = info
        # title(img_title + '. Info (bits): ' + str(info))
        # title(img_title)
        ticks = Reduce_ticks_labels(np.round(np.array(frequencies)), ticks_steps=10)
        xticks([i for i in range(len(frequencies))], ticks)
        if len(s_osc_intrinsic.avg_isi_value) == 0:
            s_osc_intrinsic.avg_isi()
        ticks = Reduce_ticks_labels(np.round(1 / s_osc_intrinsic.avg_isi_value), ticks_steps=5)
        print(ticks + [np.round(1 / s_osc_intrinsic.avg_isi_value).max()])
        yticks([i for i in range(neurons_n)], ticks+ [np.round(1 / s_osc_intrinsic.avg_isi_value).max()])
        colorbar()
        if show == True:
            plt.show()
    def plot_isi(self,average = 1, offset=0, gain=1, norm=False, label=None,
                               plot_where=None, neuron_list = [],average_percentage = None):

        if self.ismultipleidx():
            if plot_where == None:
                _, plot_where = plt.subplots(nrows=1, ncols=1)
            idx = 0
            if len(neuron_list) == 0:
                neuron_list = [i for i in range(len(self.dummyspike_collection))]
            for d_ix, dummy_structure in enumerate(self.dummyspike_collection):

                if d_ix in neuron_list:
                    dummy_structure.plot_isi(average = average, offset = offset+idx, gain = gain, norm = norm, label = label,
                                         plot_where = plot_where, average_percentage = average_percentage)
                    # idx += 1
        else:
            if len(self.isi['value']) == 0:
                self.isi_calculator(average = average)
            self.plot_function(self.isi['time'],self.isi['value'], norm = norm, gain = gain, offset = offset,
                               label = label, plot_where = plot_where)
    def plot_spikes_vs_sinusoid(self, offset, gain, norm, label, plot_where=None, return_s=False,
                                colors=[None, None]):
        '''
        This function plot the spikes in comparison with a sinusoid
        :param (boolean) norm: Flag for defining if the values should be normalized for the maximum value (used for multiple plots in 1 plot)
        :param (float) offset: Value for defining where the 0 should be in the plot (used for multiple plots in 1 plot)
        :param (float) gain: Value for defining where the maximum value should be in the plot (used for multiple plots in 1 plot)
        :param (str) label: String that can be appended to the plot for the legend (used for multiple plots in 1 plot)
        :param (boolean) return_s: Flag that defines if the resulting dummy_spikestructure should be returned or not
        :param (axis object) plot_where: Define on which figure the plot should go (useful when having multiple plots on different windows)
        :param (list) colors: Define the colors of the plot
        :return: If return_s is True the dummy_spikestructure is returned
        '''
        if len(self.sinusoid) == 0:
            print('Run first Encoding_sinusoid')
        if plot_where == None:
            plot(self.sinusoid[0], gain * self.sinusoid[1] / (self.sinusoid[1].max() * int(norm) + 1) + offset,
                 label=label, color=colors[0])
            plot(self.t, gain / 2 * self.i + offset, '.', color=colors[1],markersize=12)
        else:
            plot_where.plot(self.sinusoid[0],
                            gain * self.sinusoid[1] / (self.sinusoid[1].max() * int(norm) + 1) + offset,
                            label=label,
                            color=colors[0])
            plot_where.plot(self.t, gain / 2 * self.i + offset, '.', color=colors[1], markersize=12)
        if return_s == True:
            return self

    def plot_spikes(self, offset = 0, gain = 1, plot_where=None, return_s=False, color=None, neuron_list = []):
        '''
        The function just plot the spikes in a scatter matter.
        :param (float) offset: Value for defining where the 0 should be in the plot (used for multiple plots in 1 plot)
        :param (float) gain: Value for defining where the maximum value should be in the plot (used for multiple plots in 1 plot)
     :param (boolean) return_s: Flag that defines if the resulting dummy_spikestructure should be returned or not
        :param (axis object) plot_where: Define on which figure the plot should go (useful when having multiple plots on different windows)
        :param (color) color: Define the color of the plot
        :return: If return_s is True the dummy_spikestructure is returned
        '''
        if self.ismultipleidx():
            idx = 0
            for d_ix,dummy_structure in enumerate(self.dummyspike_collection):
                if d_ix in neuron_list:
                    dummy_structure.plot_spikes(offset=offset + idx, gain=gain,
                                                return_s=return_s,
                                                plot_where=plot_where, color=color)
                    idx += 1
        else:
            if plot_where == None:
                plot(self.t, gain / 2 * self.i + offset, '*', color=color,markersize=12)
            else:
                plot_where.plot(self.t, gain / 2 * self.i + offset, '*', color=color, markersize=12)
            if return_s == True:
                return self

    def find_highest_freqpeaks(self, minimum_valueHz=10, n_peaks=10):
        '''

        :param minimum_valueHz:
        :param n_peaks:
        :return:
        '''
        if len(self.fft) == 0:
            self.compute_fft()
        frequency_span = np.array([i * self.fs / len(self.fft) for i in range(int(len(self.fft)))])
        lower_than_xHz = np.where(frequency_span < minimum_valueHz)[0][-1]
        clean_fft = self.fft[lower_than_xHz:int(len(self.fft) / 2)]
        clean_frequency_span = frequency_span[lower_than_xHz:int(len(self.fft) / 2)]
        self.freqpeaks = sorted(
            zip(clean_fft, clean_frequency_span),
            reverse=True)[:n_peaks]
        return self.freqpeaks

    def divide_indexes(self, t, i, sinusoid=[]):
        if self.verbose:
            update_progress(progress= 0, string = 'beginning division of indexes')
        max_i = i.max()
        # array = np.unique(i)
        self.dummyspike_collection = []
        for idx,i_singular in enumerate(range(int(max_i)+1)):
            where_idx = np.where(i == i_singular)
            if len(where_idx) == 0:
                self.dummyspike_collection.append(dummy_spikestructure(t=np.NAN,i=i_singular))
            else:
                self.dummyspike_collection.append(
                    dummy_spikestructure(t=t[where_idx[0]], i=i[where_idx[0]], sinusoid=sinusoid))
            if np.mod(idx,50) == 0:
                update_progress(idx/max_i, 'division of index ' + str(i_singular)  + '/'+ str(max_i))
        if self.verbose:
            update_progress(1, 'finished division of indexes. total ' + str(max_i+1))
    def calculate_info(self,stimuli,code = 'ISI_mean', winner = 'max'):
        winner_list = []
        print(code)
        if len(self.dummyspike_collection) != len(stimuli):
            self.t = np.append(self.t,np.nan)
            self.i = np.append(self.i,len(stimuli))
            self.divide_indexes(t = self.t, i = self.i)
            assert len(self.dummyspike_collection) == len(stimuli), "Neurons are not the same number as stimuli"
        for i in np.unique(stimuli):
            collector = []
            # print(i)
            for idx in np.where([stimuli == i])[1]:
                # print(idx)
                if self.ismultipleidx():
                    spikes = self.dummyspike_collection[idx]
                else:
                    raise ValueError('You need at least two neurons to calculate info')
                if code == 'ISI_mean':
                    spikes.isi_calculator()
                    collector.append(spikes.isi['value'].mean())
                    # print(spikes.isi['value'].mean())
                elif code == 'ISI_short':
                    spikes.find_short_isi()
                    collector.append(len(spikes.isi_short))
                elif code == 'ISI_long':
                    spikes.find_long_isi()
                    collector.append(len(spikes.isi_long))
                elif code == 'Count':
                    collector.append(len(spikes.t))
                else:
                    raise ValueError( code + ': Name of code not found')
            if winner == 'max':
                winner_list.append(np.argmax(collector))
            elif winner == 'min':
                winner_list.append(np.argmin(collector))
        print(winner_list)
        stimuli_array = [[0, i] for i in range(len(np.unique(stimuli)))]
        info = calculate_information(stimuli_array,winner_list)
        return info





    def plot_count_vs_intrinsicfreqandinputfrequency(self, neurons_n, frequencies, s_osc_intrinsic, norm = False,
                                                     cmap = None, min_max = None, img_title = None):
        if len(self.count) == 0:
            self.gather_counts()
        if cmap == None:
            cmap = 'Purples'
        freq = 0
        neuron = 0
        my_avg = np.zeros([neurons_n,len(frequencies)])
        if len(self.avg_isi_value) == 0:
            self.count_in_intervals()
        for dummy_structure_ix in range(len(self.dummyspike_collection)):
            my_avg[neuron, freq] = self.count[dummy_structure_ix]
            if neuron == neurons_n - 1:
                neuron = 0
                freq += 1
            else:
                neuron += 1
        if min_max != None:
            if min_max == 'min':
                min_max_list = my_avg.min(axis=0)
            elif min_max == 'max':
                min_max_list = my_avg.max(axis=0)
            min_max_where = []
            for idx, min_max_here in enumerate(min_max_list):
                min_max_where.append(np.where(my_avg[:, idx] == min_max_here)[0][0])

            plot(min_max_where, '.', color='k', linewidth=2)
        if norm == False:
            norm = 1
        else:
            norm = my_avg.max(axis=0)
        imshow(my_avg / norm, cmap= cmap)
        # plot(min/norm)
        xlabel('Input Frequency (Hz)')
        ylabel('Intrinsic Frequency (Hz)')
        stimuli = []
        for i in range(len(frequencies)):
            stimuli += [i for j in range(neurons_n)]
        info = self.calculate_info(stimuli = stimuli,code = 'Count',winner = min_max)
        self.info[img_title] = info
        title(img_title + '. Info (bits): ' + str(info))
        ticks = Reduce_ticks_labels(np.round(np.array(frequencies)), ticks_steps=10)
        xticks([i for i in range(len(frequencies))], ticks)
        ticks = Reduce_ticks_labels(np.round(1 / s_osc_intrinsic.avg_isi_value), ticks_steps=5)
        yticks([i for i in range(neurons_n)], ticks + [np.round(1 / s_osc_intrinsic.avg_isi_value).max()])
        colorbar()
    def gather_counts(self, upper_time_intervals=[]):
        assert self.ismultipleidx() == True, "To run gather_counts() you need to have multiple indexes"
        first = True
        for dummy_structure in self.dummyspike_collection:
            if len(dummy_structure.count_in_interval) == 0:
                dummy_structure.count_in_intervals(upper_time_intervals=upper_time_intervals)
            if first == True:
                self.count = np.zeros([0, len(dummy_structure.count_in_interval)])
                first = False
            self.count = np.vstack([self.count,dummy_structure.count_in_interval])
        
    


def load_dummyspikedata_fromfile(path):
    it = np.loadtxt(path)
    s = dummy_spikestructure(t = it[0,:], i = it[1,:])
    return s

def frequency2spikes(f, Tsim):
    '''

    :param f:
    :param Tsim:
    :return:
    '''
    frequency = [f]
    neuron_array, time_array, simulation_array = generate_spikes_from_freq(
        given_frequency=frequency, duration=Tsim, break_time=0)
    s = dummy_spikestructure(t=time_array, i=neuron_array)
    return s


def create_spatialcode(stimulus_array, my_index, coding_type='WTA'):
    spatial_code = []
    orientation_value = [stimulus_array[i][1] for i in range(0, len(stimulus_array))]
    list_orientation = unique(orientation_value)
    # for column in range(0, shape(my_index)[1]):
    #     temp = ''
    # for i in my_index[:, column]:
    #     temp = temp + str(int(i))
    #     # print(temp)
    # spatial_code.append(temp)
    if coding_type == 'WTA':
        for column in range(0, shape(my_index)[1]):
            temp = ''
            counter = 0
            max_i = 0
            max_where = 0
            for i in my_index[:, column]:
                if i > max_i:
                    max_i = i
                    max_where = counter
                counter += 1
            for number in range(0, counter):
                if number == max_where:
                    temp = temp + str(1)
                else:
                    temp = temp + str(0)
                # print(temp)
            spatial_code.append(temp)
    elif coding_type == 'Spatial':
        for column in range(0, shape(my_index)[1]):
            temp = ''
            counter = 0
            max_i = 0
            max_where = 0
            for i in my_index[:, column]:
                if i > max_i:
                    max_i = i
                    max_where = counter
                counter += 1
            for i in my_index[:, column]:
                if i > max_i / 2:
                    temp = temp + str(1)
                else:
                    temp = temp + str(0)
            spatial_code.append(temp)
    elif coding_type == 'Rate':
        for column in range(0, shape(my_index)[1]):
            temp = ''
            counter = 0
            max_i = 0
            max_where = 0
            for i in my_index[:, column]:
                temp = temp + str(i)
            spatial_code.append(temp)
    else:
        print('Wrong coding type, please choose between WTA, Spatial and Temporal')
        del spatial_code
    # print(spatial_code)
    return spatial_code
def save_stimulus(data,path):
    np.savetxt(path, data)
def load_stimulus(path):
    data = np.loadtxt(path)
    return data
def statistics_on_rate(my_index, stimulus_array, sigma_collection, mean_collection, nNeurons_layer2):
    orientation_value = np.array([int(stimulus_array[i][1]) for i in range(0, len(stimulus_array))])
    list_orientation = [i for i in range(8)]
    # print(orientation_value)
    # print(list_orientation)
    value_coll = zeros([len(list_orientation), nNeurons_layer2, 100])
    sigma_array = zeros([len(list_orientation), nNeurons_layer2])
    mean_array = zeros([len(list_orientation), nNeurons_layer2])
    # print('sigma_array_dim: (' + str(len(list_orientation)) + ',' + str(nNeurons_layer2) + ")")
    for i in range(0, len(list_orientation)):
        for j in range(0, nNeurons_layer2):
            where_orientation = np.where(orientation_value == i)[0]
            # print(where_orientation)
            if len(where_orientation) > 0:
                # print(len(where_orientation))
                selected_values = my_index[j, where_orientation]
                # print(selected_values)
                sigma_array[i, j] = selected_values.std()
                mean_array[i, j] = selected_values.mean()
    sigma_collection.append(sigma_array)
    mean_collection.append(mean_array)
    return sigma_collection, mean_collection


def rate_calculator(stimulus_array, Spikes, nNeurons_layer2, orientation_adimension=True, method='Spike_Count',
                    Engine='Brian2'):
    '''
    A calculator of the average spiking rate of the different neurons of the given layer, at each stimulus presentation.
    :param stimulus_array: a list of the different input fed to the network in the form
        [[time_input0,value_input0],[time_input1,value_input1], ...]
    :param Spikes: the spiking monitor of the layer you want to analyze
    :param nNeurons_layer2: the number of neurons in that layer
    :param orientation_adimension: a variable to declare if the time_input is adimensional or not
    :param method: the method used to obtain the rate:
        - Spike_Count: Simply counts the number of spikes occured during the time of each stimulus
                    (useful for rate coding)
        - Instantaneous_avg: Computes all the instantaneous rate of the spikes and then average them along the stimulus
                    (useful for temporal coding)
    :return: a 2D NDarray made by neuron x rate_stimulus
    '''

    if get_dimensions(stimulus_array[0][0]).is_dimensionless:
        orientation_time = [stimulus_array[i][0] for i in range(0, len(stimulus_array))]
    else:
        orientation_time = [stimulus_array[i][0] / second for i in range(0, len(stimulus_array))]
    low_boundary = 0 * second
    rate = np.zeros([nNeurons_layer2, len(orientation_time)])
    counter = 0
    rate = np.zeros([nNeurons_layer2, len(orientation_time)])
    counter = 0

    if get_dimensions(stimulus_array[0][0]).is_dimensionless:
        orientation_time = [stimulus_array[i][0] for i in range(0, len(stimulus_array))]
    else:
        orientation_time = [stimulus_array[i][0] / second for i in range(0, len(stimulus_array))]

    rate = np.zeros([nNeurons_layer2, len(orientation_time)])
    counter = 0

    if Engine == 'Brian2':
        low_boundary = 0 * second
        for timestamp in orientation_time:
            if orientation_adimension == True:
                high_boundary = timestamp * second
            else:
                high_boundary = timestamp
            time_indexes = np.where((Spikes.t > low_boundary) & (Spikes.t < high_boundary))
            if len(time_indexes[0]) > 0:
                if method == 'Spike_Count':
                    for element in time_indexes[0]:
                        neuron_index = int(Spikes.it[0][element])
                        rate[(neuron_index), counter] += 1
                elif method == 'Instantaneous_avg':
                    time_here = Spikes.t[time_indexes[0]]
                    neurons_here = Spikes.i[time_indexes[0]]
                    for neuron_index in range(nNeurons_layer2):
                        where_neuron = np.where(neurons_here == neuron_index)

                        if len(where_neuron[0]) > 0:

                            time_neuron = time_here[where_neuron[0]]
                            prev = time_neuron[0]
                            my_diff = 0
                            for time in time_neuron:
                                my_diff += time - prev
                                prev = time
                            if my_diff != 0:
                                rate[neuron_index, counter] = time_neuron.size / my_diff
                        else:
                            rate[neuron_index, counter] = 0
                elif method == 'Normalized_Spike_Count':
                    for element in time_indexes[0]:
                        neuron_index = int(Spikes.it[0][element])
                        rate[(neuron_index), counter] += 1

                    rate[:, counter] = rate[:, counter] / (high_boundary - low_boundary)

                else:
                    print('Method not recognised, please choose between Spike_Count and Instantaneous_avg')
            else:
                rate[:, counter] = 0
            counter += 1
            low_boundary = high_boundary
    elif Engine == 'TouchSim':
        low_boundary = 0
        for timestamp in orientation_time:
            if orientation_adimension == True:
                high_boundary = timestamp
            else:
                high_boundary = timestamp
            time_indexes = np.where((Spikes.spikes[0] > low_boundary) & (Spikes.spikes[0] < high_boundary))
            if len(time_indexes[0]) > 0:
                if method == 'Spike_Count':
                    for element in time_indexes[0]:
                        # neuron_index = int(Spikes.it[0][element])
                        neuron_index = 0
                        rate[(neuron_index), counter] += 1
                elif method == 'Instantaneous_avg':
                    time_here = Spikes.spikes[0][time_indexes[0]]
                    # neurons_here = Spikes.i[time_indexes[0]]
                    neurons_here = 0
                    for neuron_index in range(nNeurons_layer2):
                        where_neuron = np.where(neurons_here == neuron_index)
                        if len(where_neuron[0]) > 0:
                            time_neuron = time_here[where_neuron[0]]
                            prev = time_neuron[0]
                            my_diff = 0
                            for time in time_neuron:
                                my_diff += time - prev
                                prev = time
                            if my_diff != 0:
                                rate[neuron_index, counter] = time_neuron.size / my_diff
                        else:
                            rate[neuron_index, counter] = 0
                elif method == 'Normalized_Spike_Count':
                    for element in time_indexes[0]:
                        # neuron_index = int(Spikes.it[0][element])
                        neuron_index = 0
                        rate[(neuron_index), counter] += 1
                    rate[:, counter] = rate[:, counter] / (high_boundary - low_boundary)

                    rate[:, counter] = rate[:, counter] / (high_boundary - low_boundary)

                else:
                    print('Method not recognised, please choose between Spike_Count and Instantaneous_avg')
            else:
                rate[:, counter] = 0
            counter += 1
            low_boundary = high_boundary
    elif Engine == 'TouchSim':
        low_boundary = 0
        for j, timestamp in enumerate(orientation_time):
            if orientation_adimension == True:
                high_boundary = timestamp
            else:
                high_boundary = timestamp
            for neuron in range(nNeurons_layer2):
                time_indexes = np.where(
                    (Spikes.spikes[neuron] >= low_boundary) & (Spikes.spikes[neuron] < high_boundary))
                if len(time_indexes[0]) > 0:
                    if method == 'Spike_Count':
                        rate[neuron, j] = len(time_indexes)
                    elif method == 'Instantaneous_avg':
                        time_here = Spikes.spikes[neuron][time_indexes[0]]
                        # neurons_here = Spikes.i[time_indexes[0]]
                        time_neuron = 0
                        prev = time_neuron[0]
                        my_diff = 0
                        for time in time_neuron:
                            my_diff += time - prev
                            prev = time
                            if my_diff != 0:
                                rate[neuron, j] = time_neuron.size / my_diff
                        else:
                            rate[neuron_index, j] = 0
                    elif method == 'Normalized_Spike_Count':
                        for element in time_indexes[0]:
                            # neuron_index = int(Spikes.it[0][element])
                            neuron_index = 0
                            rate[(neuron_index), j] += 1
                        rate[:, j] = rate[:, counter] / (high_boundary - low_boundary)

                    else:
                        print('Method not recognised, please choose between Spike_Count and Instantaneous_avg')
            low_boundary = high_boundary
    return rate


def RemoveSpikesTooClose(time_array, neuron_array):
    indexes = np.diff(time_array) < 1e-3
    while (indexes.max() == True):
        time_array = np.delete(time_array, np.append(False, indexes))
        neuron_array = np.delete(neuron_array, np.append(False, indexes))
        indexes = np.diff(time_array) < 1e-3
    return time_array, neuron_array


def generate_spikes_from_freq(given_frequency, duration=1, break_time=0.1, time_array=None, neuron_array=None,trials_per_stimulus = 1, noise = False):
    break_time = break_time * second
    if (time_array == None) & (neuron_array == None):
        time_array = np.array([])
        neuron_array = np.array([])
    simulation_array = []
    time_now = 0 * second
    for ix in range(len(given_frequency)):
        for trial in range(trials_per_stimulus):
            if isinstance(duration, list):
                if ix == 0:
                    duration_list = duration
                duration = duration_list[ix]
            time_array_here = np.zeros([int(duration * given_frequency[ix])])
            neuron_array_here = np.zeros([int(duration * given_frequency[ix])])
            for step in range(int(duration * given_frequency[ix])):
                time_array_here[step] = time_now + 1 / given_frequency[ix] * step * second
            time_array = np.append(time_array, time_array_here)
            neuron_array = np.append(neuron_array, neuron_array_here+ix*trials_per_stimulus+trial)
        # time_now += duration * second + break_time
        simulation_array.append([(time_now) / second, 0])
        simulation_array.append([(time_now) / second, 0])
    avg_distance = np.mean(np.diff(time_array))
    if noise == True:
        noise = np.random.normal(-1/(50*100), 1/(50*100), len(time_array))
        time_array_noisy_neg = time_array+noise
        time_array_noisy = time_array_noisy_neg[time_array_noisy_neg > 0]
        # neuron_array = neuron_array[time_array_noisy_neg > 0]
        # print(noise)
        return neuron_array, time_array, simulation_array
    else:
        return neuron_array, time_array, simulation_array




def From_Rate_to_ConfusionMatrix(rate_in, rate_out, levels, neurons_n):
    intervals_input, ticks = search_interval(rate=rate_in, levels=levels, ticks=True)
    winner, multi = find_winner(rate_out, allow_half_winners=False, ignore_lower=True)
    confusion_matrix = create_confusion_matrix(stimulus=intervals_input, response=winner, levels_stimulus=levels,
                                               levels_response=neurons_n, embedded_fails=True, allow_half_winners=False)
    return confusion_matrix


def find_winner(rate, allow_half_winners=False, ignore_lower=False):
    index_out = rate
    winner = np.zeros([index_out.shape[1]])
    multiple_winning = 0
    for row in range(index_out.shape[1]):
        winner_here = np.where(index_out[:, row] == index_out[:, row].max())
        if ignore_lower == True:
            winner_here = np.where(index_out[1:, row] == index_out[1:, row].max())

        if len(winner_here[0]) > 1:
            if (allow_half_winners == True) & (len(winner_here[0]) == 2):
                if (np.abs(winner_here[0][0] - winner_here[0][1]) == 1):
                    winner[row] = (winner_here[0][0] + winner_here[0][1]) / 2
            else:
                multiple_winning += 1
                winner[row] = np.NaN
        else:
            try:
                winner[row] = winner_here[0][0] + 1
            except:
                print('ueue')
    return winner, multiple_winning


def FromTouchSimtoEventDriven(Spikes, specific_neurons=None, time_array=None, neuron_array=None, time_now=0):
    import numpy as np
    if (time_array.any() == None) & (neuron_array.any() == None):
        time_array = np.array([])
        neuron_array = np.array([])
    if specific_neurons == None:
        for i in range(np.size(Spikes)):
            time_array = np.append(time_array, Spikes[i] + time_now)
            neuron_array = np.append(neuron_array, np.array([i for j in range(len(Spikes[i]))]))
    else:
        for idx, neuron in enumerate(specific_neurons):
            time_array = np.append(time_array, Spikes[neuron] + time_now)
            neuron_array = np.append(neuron_array, np.array([idx for j in range(len(Spikes[neuron]))]))
    sort_indexes = argsort(time_array)
    time_array = time_array[sort_indexes]
    neuron_array = neuron_array[sort_indexes]
    return time_array, neuron_array


def search_interval(rate, levels, ticks=False, offset=True, force_interval=None):
    lsb0 = False
    if isinstance(rate, tuple):
        rate = rate[0]

    if force_interval == None:
        max_value = rate.max()
        offset_value = rate.min() * (0 + int(offset))
    else:
        offset_value = force_interval[0]
        max_value = force_interval[1]
    lsb = (max_value - offset_value) / (levels - 1)
    if len(rate) == 1:
        lsb = rate[0]
    elif lsb == 0:
        print('LSB is 0!')
        lsb0 = True
        print(rate)

    discrete_elements = np.zeros([rate.shape[0]])
    for time in range(rate.shape[0]):
        if lsb0 == False:
            try:
                value = int(np.floor((rate[time] - offset_value) / lsb))
            except ValueError:
                print('Error!')

        discrete_elements[time] = value
    ticks_values = []
    if ticks == True:
        for level in range(levels):
            ticks_values.append(round(offset_value + lsb * level))
        return discrete_elements, ticks_values
    else:
        return discrete_elements


def create_confusion_matrix(stimulus, response, levels_stimulus=10, levels_response=10, embedded_fails=False,
                            allow_half_winners=False):
    if embedded_fails == False:
        confusion_matrix = np.zeros([levels_stimulus, levels_response + levels_response * int(allow_half_winners)])
        Nan_matrix = np.zeros([levels_stimulus, 1])
    else:
        confusion_matrix = np.zeros(
            [levels_stimulus, levels_response + levels_response * int(allow_half_winners) + 1])

    for row in range(len(response)):
        if isnan(response[row]) == False:
            confusion_matrix[int(stimulus[row]), int(response[row] + response[row] * int(allow_half_winners))] += 1
        else:
            if embedded_fails == False:
                Nan_matrix[int(stimulus[row]), 0] += 1
            else:
                confusion_matrix[
                    int(stimulus[row]), levels_response + levels_response * int(allow_half_winners)] -= 1
    if embedded_fails == False:
        return confusion_matrix, Nan_matrix
    elif allow_half_winners == True:
        ticks = []
        for number in range(levels_response * 2):
            ticks.append(number / 2)
        return confusion_matrix, ticks
    else:
        return confusion_matrix


def Isi_calculator(spikes, neuron_list, axis_point=0, average_n=1, marker=False, sweeping_var=None,
                   plot=True, hist=False):
    '''
    This function computes the interspike interval of 1 or several spike trains
    :param spikes: A Brian2 spike monitor
    :param neuron_list: the neurons in the spike monitors that need ISI
    :param axis_point: Where to plot the ISI graph (if plot == True)
    :param average_n: Averaging value of the ISI
    :param marker: Put marker at the spike time (if plot == True)
    :param sweeping_var: A variable we sweeped
    :param plot: If True the function comes out with a plot
    :param hist: if True the function results in a histogram
    :return: return a list with size len(neuron_list) of 2 ndarrays: interspike intervals and time of spike
    '''
    if (isinstance(axis_point, int)) & (plot == True):
        fig_point, axis_point = subplots(nrows=1, ncols=1)
    # n_neurons = len(list(set(spikes.i)))
    isi_collection = []
    temp2 = []
    for i in neuron_list:
        temp = []
        temp_t = []
        isi_average = np.array([])
        isi_t_average = np.array([])
        spikes_ix = np.where(spikes.i == i)[0]

        bins = 100
        sum = 0
        selected_spikes = spikes.t[spikes_ix]
        time_diff = np.diff(selected_spikes)
        if average_n != 1:
            steps = int(np.floor(len(time_diff) / average_n))
            isi_t_average = np.append(isi_t_average, 0)
            for step in range(steps - 1):
                isi_average = np.append(isi_average,
                                        1 / (np.average(time_diff[step * average_n:(step + 1) * average_n])))
                isi_t_average = np.append(isi_t_average,
                                          np.average(selected_spikes[step * average_n:(step + 1) * average_n]))
        else:
            isi_t_average = selected_spikes
            isi_average = 1 / time_diff
        isi_collection_temp = np.vstack([isi_t_average[:-1] / second, isi_average / Hz])
        # isi_collection = np.zeros([2, 0])
        isi_collection.append(isi_collection_temp)
        if marker == True:
            axis_point.plot(isi_t_average[1:], isi_average, marker='.')
        elif plot == True:
            axis_point.plot(isi_t_average[1:], isi_average)
            # axis_point.set_ylim([100,105])
        elif hist == True:
            try:
                if spikes.t[spikes_ix[0][-1]] - spikes.t[spikes_ix[0][0]] > 0:
                    temp2.append(spikes.count[i] / (spikes.t[spikes_ix[0][-1]] - spikes.t[spikes_ix[0][0]]))
                else:
                    temp2.append(0 * ms)
            except IndexError:
                temp2.append(0 * ms)
    if hist == 1:
        axis_point.plot(sweeping_var, temp2)
    return isi_collection


def compute_spikes_ffts(spikes, neurons_list, stimulus_array, fft_bins, visualization, duration, fs,
                        return_result=False, Engine='Brian2'):
    if Engine == 'TouchSim':
        if get_dimensions(fs).is_dimensionless == False:
            fs = fs / Hz
    neuron_collection = []
    for neuron in neurons_list:
        sampling_array = [[i / fs, 0] for i in range(int(fs * duration))]
        Rate = rate_calculator(stimulus_array=sampling_array, Spikes=spikes, nNeurons_layer2=len(neurons_list),
                               method='Spike_Count', Engine=Engine)[0]
        freq_coll = compute_fft_for_each_stimulus(Rate, fs, stimulus_array, fft_bins, duration=duration)
        neuron_collection.append(freq_coll)
    if visualization == 'fft':
        if plot == True:
            visualize_fft()
    elif visualization == 'spectrogram':
        if plot == True:
            visualize_spectrogram()
    if return_result == True:
        return freq_coll, Rate


def compute_fft_for_each_stimulus(Rate, fs, stimulus_array, fft_bins, duration):
    prev = 0
    freq_coll = []
    stimulus = 0
    analog_clock = 1 / fs
    for i in range(len(Rate)):
        try:
            if analog_clock * (i + 1) > stimulus_array[stimulus][0]:
                freq_coll.append(np.abs(fft(Rate[prev:i], n=fft_bins)))
                prev = i
                stimulus += 1
        except IndexError:
            print(i)
    freq_coll.append(np.abs(fft(Rate[prev:i], n=fft_bins)))
    return freq_coll


def visualize_spectrogram(freq_coll, axis_plot, fs):
    ffts = np.abs(freq_coll)
    axis_plot.imshow(ffts.T, aspect='auto')
    ylim([0, fs / 2])


def visualize_fft(axis_plot):
    if isinstance(axis_plot, int):
        fig_plot, axis_plot = plt.subplots(nrows=len(neuron_list), ncols=len(uniques))
    for n_ix, freq_coll in enumerate(neuron_collection):
        for i, fft_here in enumerate(freq_coll):
            which_plot = uniques.index(stimulus_array[i][1][3])

            if isinstance(axis_plot, int) == True:
                if len(uniques) > 1:
                    axis_plot[n_ix][which_plot].plot([i * fs for i in range(int(len(fft_abs) / 2))],
                                                     fft_abs[0:int(len(fft_abs) / 2)], color=plot_color)
                    axis_plot[n_ix][which_plot].set_xlim([0, fs / 2])
                else:
                    axis_plot[n_ix].plot([i * fs / n for i in range(int(n / 2))], fft_abs[0:int(n / 2)],
                                         color=plot_color)
                    axis_plot[n_ix].set_xlim([0, fs / 2])
            else:
                if len(uniques) > 1:
                    axis_plot[which_plot].plot([i * fs / n for i in range(int(n / 2))], fft_abs[0:int(n / 2)],
                                               color=plot_color)
                    axis_plot[which_plot].set_xlim([0, fs / 2])
                else:
                    axis_plot.plot([i * fs for i in range(int(len(fft_abs) / 2))], fft_abs[0:int(len(fft_abs) / 2)],
                                   color=plot_color)
                    axis_plot.set_xlim([0, fs / 2])


def visualize_spectrogram(freq_coll, axis_plot, fs):
    '''
    This function visualizes the spectrogram of the ffts
    :param freq_coll: the ffts
    :param axis_plot: where to plot
    :param fs: sampling frequency
    :return: None
    '''
    ffts = np.abs(freq_coll)
    axis_plot.imshow(ffts.T, aspect='auto')
    ylim([0, fs / 2])


def compute_fft_for_each_stimulus(Rate, fs, stimulus_array, fft_bins, duration):
    '''
    This function computes the fft of a specific stimulus
    :param Rate: A spike train converted into spike rate vector
    :param fs: the sampling frequency of the input
    :param stimulus_array: The information on when each batch finishes [[time1,stim1],[time2,stim2],..]
    :param fft_bins: The bins required by the FFT
    :param duration: how small each bins of time for rate calculation should be
    :return: it returns the fft
    '''
    prev = 0
    freq_coll = []
    stimulus = 0
    analog_clock = 1 / fs
    if fft_bins == None:
        fft_bins = int(fs * duration)
    for i in range(len(Rate)):
        try:
            if analog_clock * (i + 1) > stimulus_array[stimulus][0]:
                freq_space = fftfreq(fft_bins, d=analog_clock)
                freq_space_halfspectrum = freq_space[:int(fft_bins / 2)]
                myfft = np.abs(fft(Rate[prev:i], n=fft_bins))
                freq_coll.append([myfft[:int(len(myfft) / 2)], freq_space_halfspectrum])



                prev = i
                stimulus += 1
        except IndexError:
            print(i)
    freq_space = fftfreq(fft_bins, d=analog_clock)
    freq_space_halfspectrum = freq_space[:int(fft_bins / 2)]
    myfft = np.abs(fft(Rate[prev:i], n=fft_bins))
    freq_coll.append([myfft[:int(len(myfft) / 2)], freq_space_halfspectrum])
    return freq_coll


def compute_spikes_ffts(spikes, neurons_list, stimulus_array, fft_bins, visualization,
                        duration, fs, return_result=False, Engine='Brian2', Wanna_plot=False):
    '''
    This fuction computes Fast Fourier Transform of spike trains.
    :param spikes: Spike Monitor (Brian2 or TouchSim)
    :param neurons_list: The list of the neurons in the spike monitor that require FFT
    :param stimulus_array: The information on when each batch finishes [[time1,stim1],[time2,stim2],..]
    :param fft_bins: The bins required by the FFT
    :param visualization: 'fft' or 'spectrogram' defines how you want to visualize the result
    :param duration: how small each bins of time for rate calculation should be
    :param fs: the sampling frequency of the input
    :param return_result: if True returns the calculated ffts
    :param Engine: define if you are using Brian2 or TouchSim
    :param Wanna_plot: If True it invokes the function plot_fft or plot_spectrogram
    :return: if return_result is True returns the calculated ffts and the rate
    '''
    if Engine == 'TouchSim':
        if get_dimensions(fs).is_dimensionless == False:
            fs = fs / Hz
    neuron_collection = []

    sampling_array = [[np.min(spikes.t) + i / fs, 0] for i in range(int(fs * np.max(spikes.t)))]
    Rate = rate_calculator(stimulus_array=sampling_array, Spikes=spikes, nNeurons_layer2=len(neurons_list),
                           method='Spike_Count', Engine=Engine)
    for neuron in range(len(neurons_list)):
        freq_coll = compute_fft_for_each_stimulus(Rate[neuron, :], fs, stimulus_array, fft_bins, duration=duration)
        neuron_collection.append(freq_coll)
    if visualization == 'fft':
        if Wanna_plot == True:
            visualize_fft()
    elif visualization == 'spectrogram':
        if Wanna_plot == True:
            visualize_spectrogram()
    if return_result == True:
        return neuron_collection, Rate


def plot_fft(analog_signal, n_neurons, stimulus_array, n=100, visualization='spectrogram', analog_clock=1 * ms,
             axis_plot=0, type='monitor', neuron_list=[], plot_color=None, sort_stimuli=False, return_result=False,
             Engine='Brian2', duration=None):
    fs = 1 / analog_clock

    if Engine == 'TouchSim':
        if get_dimensions(fs).is_dimensionless == False:
            fs = fs / Hz

    velocities = [stimulus_array[i][1][3] for i in range(len(stimulus_array))]
    uniques = list(set(velocities))
    if sort_stimuli == True:
        uniques.sort()
    if isinstance(axis_plot, int):
        if visualization == 'fft':
            fig_plot, axis_plot = plt.subplots(nrows=1, ncols=len(uniques))
        elif visualization == 'spectrogram':
            fig_plot, axis_plot = plt.subplots(nrows=1, ncols=1)
    if type == 'monitor':
        prev = 0
        freq_coll = []
        stimulus = 0
        for i in range(len(analog_signal)):
            if analog_clock * (i + 1) > stimulus_array[stimulus][0] * second:
                # freq_coll.append(
                #     fft(butter_lowpass_filter(analog_signal[prev:i], cutoff=fs / 2 - 1, fs=fs, order=3), n=n))
                freq_coll.append(fft(analog_signal[prev:i]))
                prev = i
                stimulus += 1
        # freq_coll.append(fft(butter_lowpass_filter(analog_signal[prev:], cutoff=fs / 2 - 1, fs=fs, order=3), n=n))
        freq_coll.append(fft(analog_signal[prev:]))

        if visualization == 'spectrogram':
            ffts = np.abs(freq_coll)
            axis_plot.imshow(ffts.T, aspect='auto')
            ylim([0, fs / 2])
        elif visualization == 'fft':

            for i, fft_here in enumerate(freq_coll):
                which_plot = uniques.index(stimulus_array[i][1][3])
                fft_abs = np.abs(fft_here)
                if len(uniques) > 1:
                    axis_plot[which_plot].plot([i * fs / n for i in range(int(n / 2))], fft_abs[0:int(n / 2)],
                                               color=plot_color)
                    axis_plot[which_plot].set_xlim([0, fs / 2])
                    axis_plot[which_plot].set_title(str(np.round(1000 / stimulus_array[i][1][3])) + '/mm')
                else:
                    axis_plot.plot([i * fs / n for i in range(int(n / 2))], fft_abs[0:int(n / 2)], color=plot_color)
                    axis_plot.set_xlim([0, fs / 2])
                    axis_plot.set_title(str(np.round(1000 / stimulus_array[i][1][3])) + '/mm')
            # fig_plot.suptitle('Spike Freq vs Space Freq')
    elif type == 'time_spikes':
        steps = np.array([i / fs for i in range(int(analog_signal.max() * fs))])
        counters = np.zeros([len(steps)])
        for time in analog_signal:
            for step in steps:
                index = np.where(time > steps)
            counters[index] = 1
        analog_signal = counters
        del counters
        prev = 0
        freq_coll = []
        stimulus = 0
        for i in range(len(analog_signal)):
            if analog_clock * (i + 1) > stimulus_array[stimulus][0] * second:
                freq_coll.append(
                    fft(analog_signal[prev:i], n=n))
                prev = i
                stimulus += 1
        freq_coll.append(fft(analog_signal[prev:], n=n))
        if visualization == 'spectrogram':
            ffts = np.abs(freq_coll)
            axis_plot.imshow(ffts.T, aspect='auto')
            ylim([0, fs / 2])
        elif visualization == 'fft':

            for i, fft_here in enumerate(freq_coll):
                which_plot = uniques.index(stimulus_array[i][1][3])
                fft_abs = np.abs(fft_here)
                if len(uniques) > 1:
                    axis_plot[which_plot].plot([i * fs / n for i in range(int(n / 2))], fft_abs[0:int(n / 2)],
                                               color=plot_color)
                    axis_plot[which_plot].set_xlim([0, fs / 2])
                else:
                    axis_plot.plot([i * fs / n for i in range(int(n / 2))], fft_abs[0:int(n / 2)], color=plot_color)
                    axis_plot.set_xlim([0, fs / 2])
    elif type == 'spikeMonitor':
        myfft, rate = compute_spikes_ffts(spikes=analog_signal, neurons_list=neuron_list, stimulus_array=stimulus_array,
                                          fft_bins=n, visualization=visualization, duration=duration,
                                          fs=1 / analog_clock,
                                          return_result=return_result, Engine=Engine)
        return myfft, rate
    else:
        print('Please select a type between monitor,time_spikes and spikeMonitor')
        raise EnvironmentError
