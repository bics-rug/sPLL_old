
import Simulation_parameters
import os, sys
from brian2 import *

code_name = 'PLL_test'


Current_PATH = os.getcwd()
FILE_PATH = os.path.dirname(os.getcwd())
CODE_PATH = os.path.join(FILE_PATH, 'Code')

sys.path.append(os.path.join(CODE_PATH,'utils'))
from utils_plot import *
from utils_spikes import *
from utils_misc import *
Data_PATH = create_folder(os.path.join(FILE_PATH,'Data'))
PRECOMPILED = create_folder(os.path.join(FILE_PATH,'Precompiled'))
IMAGE_PATH =  create_folder(os.path.join(FILE_PATH,'Images'))
s_osc_intrinsic = load_dummyspikedata_fromfile(os.path.join(Data_PATH, 's_osc_intrinsic_freq' + '.csv'))
s_tde = load_dummyspikedata_fromfile(os.path.join(Data_PATH, code_name + '_' +  's_tde' + '.csv'))
# s_osc = load_dummyspikedata_fromfile(os.path.join(Data_PATH, code_name + '_' +  's_osc' + '.csv'))
s_in = load_dummyspikedata_fromfile(os.path.join(Data_PATH, code_name + '_' +  's_in' + '.csv'))

# s_in.plot_fft(fs = 100e3, max_x=500*hertz,min_x=0.1*hertz, gain = 5, norm = True)
# figure()
# print('plotting')
# s_osc.plot_fft(fs = 10e3, max_x=500*hertz,min_x=0.1*hertz, gain = 10, norm = True)
# show()
print('')

# s_nWTA = load_dummyspikedata_fromfile(os.path.join(Data_PATH, 's_wta' + '.csv'))
# s_hp2 = load_dummyspikedata_fromfile(os.path.join(Data_PATH, 's_hp2.csv'))
# s_in = load_dummyspikedata_fromfile(os.path.join(Data_PATH, 's_in.csv'))
neurons_n = 20
frequencies = [i for i in linspace(50,200,50)]
# s_osc_intrinsic.avg_isi()
# stimuli = []
# for i in range(len(frequencies)):
#     stimuli += [i for j in range(neurons_n)]
# stimuli_2 = [i for i in range(len(frequencies))]
# s_in.calculate_info(stimuli = stimuli_2)

# s_tde.calculate_info(stimuli = stimuli)
# s_tde.plot_isi_vs_intrinsicfreqandinputfrequency(neurons_n, frequencies, s_osc_intrinsic, norm = True, img_title = 'ISI_mean (normalized per column)')
# figure()
# s_tde.plot_isilong_vs_intrinsicfreqandinputfrequency(neurons_n, frequencies, s_osc_intrinsic, norm = True, min_max = 'max', img_title = 'ISI_long_max', cmap = 'Greens')
# figure()
# s_tde.plot_isishort_vs_intrinsicfreqandinputfrequency(neurons_n, frequencies, s_osc_intrinsic, norm = True, min_max = 'max', img_title = 'ISI_short_max', cmap = 'Blues')
# figure()
s_tde.plot_count_vs_intrinsicfreqandinputfrequency(neurons_n,frequencies,s_osc_intrinsic, norm = True, min_max = 'max', img_title = 'Count max', cmap = 'Reds')
#
# figure()
# s_tde.plot_isi_vs_intrinsicfreqandinputfrequency(neurons_n, frequencies, s_osc_intrinsic, norm = True, min_max = 'min', img_title = 'ISI_mean_min', cmap = 'GnBu')
# figure()
# s_tde.plot_isilong_vs_intrinsicfreqandinputfrequency(neurons_n, frequencies, s_osc_intrinsic, norm = True, min_max = 'min', img_title = 'ISI_long_min', cmap = 'PuBu')
# figure()
# s_tde.plot_isishort_vs_intrinsicfreqandinputfrequency(neurons_n, frequencies, s_osc_intrinsic, norm = True, min_max = 'min', img_title = 'ISI_short_min', cmap = 'PuRd')
# figure()
# s_tde.plot_count_vs_intrinsicfreqandinputfrequency(neurons_n,frequencies,s_osc_intrinsic, norm = True, min_max = 'min', img_title = 'Count min', cmap = 'RdPu')
show()
# figure()
# s_hp2.plot_isi_vs_intrinsicfreqandinputfrequency(neurons_n, frequencies, s_osc_intrinsic, norm = True, min_max = 'max', img_title = 'ISI_mean_max')
# figure()
# s_hp2.plot_isilong_vs_intrinsicfreqandinputfrequency(neurons_n, frequencies, s_osc_intrinsic, norm = True, min_max = 'max', img_title = 'ISI_long_min', cmap = 'PuBu')
# figure()
# s_hp2.plot_isishort_vs_intrinsicfreqandinputfrequency(neurons_n, frequencies, s_osc_intrinsic, norm = True, min_max = 'max', img_title = 'ISI_short_max', cmap = 'Blues')
# figure()
# s_hp2.plot_count_vs_intrinsicfreqandinputfrequency(neurons_n,frequencies,s_osc_intrinsic, norm = True, min_max = 'max', img_title = 'Count max', cmap = 'Reds')
# #
