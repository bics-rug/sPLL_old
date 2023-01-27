import csv
import matplotlib.pyplot as plt
import os
import sys
from brian2 import *
from matplotlib import cm as mycm

import Simulation_parameters
from utils.utils_spikes import dummy_spikestructure, generate_spikes_from_freq
from utils.utils_misc import create_folder, create_image_folder


code_name = 'PLL_test'
Current_PATH = os.getcwd()
FILE_PATH = os.path.dirname(os.getcwd())
CODE_PATH = os.path.join(FILE_PATH, 'Code')
sys.path.append(os.path.join(CODE_PATH,'utils'))

DATA_PATH = create_folder(os.path.join(FILE_PATH, 'Data'))
PRECOMPILED = create_folder(os.path.join(FILE_PATH, 'Precompiled'))
IMAGE_PATH =  create_folder(os.path.join(FILE_PATH, 'Images'))


def get_oscillator_neuron(n):
    Oscillator_model = Equations('''
                                    dv/dt = -v/(tau_Osc) + (I + Ie - Ii)/(0.01*pF): volt
                                    I : amp
                                    dIe/dt = -Ie/(tau_Osc_Ie) : amp
                                    dIi/dt = -Ii/(tau_Osc_Ii) : amp
                                    ''')
    Oscillator_neuron = NeuronGroup(n, model=Oscillator_model, method='euler', threshold='v > v_Osc_threshold',
                                   refractory='refrac_Osc',
                                   reset='v = 0*mV')
    return Oscillator_neuron


def create_PLL(input_n,neurons_n, neuron_array, time_array):

    Fake_Neuron = SpikeGeneratorGroup(input_n, neuron_array, time_array * second, name='spike_gen')

    TDE_model = Equations('''dv/dt = -v/(tau_TDE) +Ie_NMDA_h/(0.01*pF) : volt
                             dIe_h/dt = -Ie_h/(tau_TDE_Ie_h) : amp
                             dIe_NMDA_h/dt = -Ie_NMDA_h/(tau_TDE_Ie_NMDA_h) : amp
                             ''')
    TDE_Neuron_plus = NeuronGroup(neurons_n*input_n, model=TDE_model, method='euler', 
                                  threshold='v > v_TDE_threshold',
                                  refractory='refrac_TDE',
                                  reset='v = 0*mV')

    Oscillator_neuron = get_oscillator_neuron(neurons_n*input_n)

    HP_model = Equations('''dv/dt = -v/(tau_HP) + (I + Ie - Ii)/(0.01*pF): volt
                            I : amp
                            dIe/dt = -Ie/(tau_HP_Ie) : amp
                            dIi/dt = -Ii/(tau_HP_Ii) : amp
                            ''')

    HP0 = NeuronGroup(neurons_n*input_n, model=HP_model, method='euler', 
                      threshold='v > v_HP_threshold', refractory='refrac_HP',
                      reset='v = 0*mV')
    HP1 = NeuronGroup(neurons_n*input_n, model=HP_model, method='euler', 
                      threshold='v > v_HP_threshold', refractory='refrac_HP',
                      reset='v = 0*mV')
    HP2 = NeuronGroup(neurons_n*input_n, model=HP_model, method='euler', 
                      threshold='v > v_HP_threshold', refractory='refrac_HP',
                      reset='v = 0*mV')

    WTA_model = Equations('''dv/dt = -v/(tau_WTA) + (I + Ie - Ii)/(0.01*pF) : volt
                             I : amp
                             dIe/dt = -(Ie)/(tau_WTA_Ie) : amp
                             dIi/dt = -Ii/(tau_WTA_Ii) : amp
                          ''')

    nWTA = NeuronGroup(neurons_n*input_n, model=WTA_model, method='euler', 
                       threshold='v > v_WTA_threshold', refractory='refrac_WTA',
                       reset='v = 0*mV')
    nWTA.I = 'WTA_stable_current'
    
    Poisson_to_nWTA = PoissonGroup(input_n, 200 * Hz)
    
    gWTA = NeuronGroup(input_n, model=WTA_model, method='euler', 
                       threshold='v > v_WTA_threshold', refractory='refrac_WTA',
                       reset='v = 0*mV')

    return (
        Fake_Neuron, Oscillator_neuron, TDE_Neuron_plus,
        HP0, HP1, HP2, nWTA, gWTA, Poisson_to_nWTA
    )


def generate_NMDA_matrix(Fake_Neuron,TDE):
    matrix_of_synapses = np.zeros([len(Fake_Neuron),len(TDE)])
    delta_thr = int(len(TDE) / len(Fake_Neuron))
    low_thr = 0
    from_ix = 0
    from_ix_matrix = np.array([])
    to_ix_matrix = np.array([])
    for to_ix in range(len(TDE)):
        # print('to_ix' + str(to_ix))
        if to_ix < delta_thr + low_thr:
            matrix_of_synapses[from_ix,to_ix] = 1  # set up connection for "trigger"
            from_ix_matrix = np.append(from_ix_matrix,from_ix)
            to_ix_matrix = np.append(to_ix_matrix,to_ix)
            # print('Connecting from: ' + str(from_ix) + "to: " + str(to_ix))
        else:
            low_thr = delta_thr + low_thr
            from_ix += 1
            matrix_of_synapses[from_ix,to_ix] = 1
            from_ix_matrix = np.append(from_ix_matrix,from_ix)
            to_ix_matrix = np.append(to_ix_matrix,to_ix)
    return from_ix_matrix.astype(int),to_ix_matrix.astype(int)


def create_synapses(Fake_Neuron, Oscillator_neuron, TDE, HP0, HP1, HP2, nWTA, gWTA, Poisson_to_nWTA):

    NMDA_h_plus = Synapses(Fake_Neuron, TDE, on_pre='Ie_NMDA_h = qe_NMDA * int(Ie_h>(Ie_th)) * Ie_h')
    # NMDA_h_plus.w = matrix_of_synapses
    from_ix_matrix, to_ix_matrix = generate_NMDA_matrix(Fake_Neuron,TDE)
    NMDA_h_plus.connect(i = from_ix_matrix, j = to_ix_matrix)

    # NMDA_h_plus
    # set up connection for "trigger"
    S_h_plus = Synapses(Oscillator_neuron, TDE, on_pre='Ie_h = qe')
    S_h_plus.connect('i == j')  # set up connection for "facilatory"

    FB_syn = Synapses(TDE, Oscillator_neuron, on_pre='Ie_post += TDE_to_Osc_current')
    FB_syn.connect('i == j')

    HP0_syn = Synapses(TDE, HP0, on_pre='Ie_post +=TDE_to_HP0_current')
    HP0_syn.connect('i == j')

    HP1_syn = Synapses(HP0, HP1, on_pre='Ie_post +=HP0_to_HP1_current')
    HP1_syn.connect('i == j')

    HP2_syn = Synapses(HP1, HP2, on_pre='Ie_post +=HP1_to_HP2_current')
    HP2_syn.connect('i == j')



    to_nWTA = Synapses(HP2, nWTA, on_pre='Ii_post += HP2_to_nWTA_current')
    to_nWTA.connect('i == j')
    to_ix_matrix, from_ix_matrix = generate_NMDA_matrix(gWTA, nWTA)
    to_gWTA = Synapses(nWTA, gWTA, on_pre='Ie_post += nWTA_to_gWTA_current')
    # to_gWTA.connect(i = from_ix_matrix, j = to_ix_matrix)

    Poisson_syn = Synapses(Poisson_to_nWTA, nWTA, on_pre='Ie_post += 10*pA')
    Poisson_syn.connect(i = to_ix_matrix, j = from_ix_matrix)
    
    from_gWTA = Synapses(gWTA, nWTA, on_pre='Ii_post += gWTA_to_nWTA_current')
    # from_gWTA.connect(i=to_ix_matrix, j=from_ix_matrix)
    normalize_WTA = Synapses(Oscillator_neuron,nWTA,on_pre='Ie_post += 100*pA')
    # normalize_WTA.connect(i=from_ix_matrix,j=to_ix_matrix)
    
    return (
        NMDA_h_plus, S_h_plus, FB_syn, HP0_syn, HP1_syn, HP2_syn,
        to_nWTA, from_gWTA, to_gWTA, Poisson_syn, normalize_WTA
    )


def create_monitors(Fake_Neuron, Oscillator_neuron, nWTA, HP0, HP1, HP2, TDE, gWTA):
    Fake_spikes = SpikeMonitor(Fake_Neuron)
    Osc_spikes = SpikeMonitor(Oscillator_neuron)
    nWTA_spikes = SpikeMonitor(nWTA)
    HP0_spikes = SpikeMonitor(HP0)
    HP1_spikes = SpikeMonitor(HP1)
    HP2_spikes = SpikeMonitor(HP2)
    TDE_spikes = SpikeMonitor(TDE)
    Gwta_spikes = SpikeMonitor(gWTA)
    return (
        Fake_spikes, Osc_spikes, nWTA_spikes, HP0_spikes, HP1_spikes, HP2_spikes, 
        TDE_spikes, Gwta_spikes, TDE_spikes
    )


def run_the_script(data):
    index = data[0]
    parameters = data[1]
    # print("The transmitted parameters are " + str(parameters['spatial_frequency']))
    output = recognise_pattern(index, parameters)
    return output


def Test_oscillator(n):
    Oscillator_Neuron = get_oscillator_neuron(n)
    osc_monitor = SpikeMonitor(Oscillator_Neuron)
    defaultclock.dt = parameters['here_clock']
    for neuron_idx in range(parameters['neurons_n']):
        Oscillator_Neuron.I[neuron_idx] = parameters['I_minimum_osc'] + parameters['I_step_osc'] * neuron_idx
    print('starting simulation for getting intrisic frequency of Oscillator Neurons')
    run(1000*ms, report = 'stdout')
    s_osc = dummy_spikestructure(t=osc_monitor.t, i=osc_monitor.i)
    s_osc.avg_isi()
    return s_osc


def recognise_pattern(index, parameters):
    print('creating frequencies')
    # Here we import the analog signals and we create spike out of it
    fs = second / parameters['analog_clock']

    s_osc_intrinsic_freq = Test_oscillator(parameters['neurons_n'])
    s_osc_intrinsic_freq.save_data(path=os.path.join(DATA_PATH, 's_osc_intrinsic_freq' + '.csv'))

    # s_input = frequency2spikes(f = parameters['frequencies'][index],Tsim = 1)
    overall_neuron_array = []
    overall_time_array = []
    if type(parameters['frequencies'][0]) == list:
        encoding_multi = True
    else:
        encoding_multi = False
    for f_ix,freq in enumerate(parameters['frequencies']):
        if encoding_multi:
            time_array, neuron_array, simulation_array = enconding_multiple_frequencies(freq)
        else:
            neuron_array,time_array,simulation_array = generate_spikes_from_freq(given_frequency = [freq])
        overall_neuron_array = np.append(overall_neuron_array,neuron_array+f_ix)
        overall_time_array = np.append(overall_time_array,time_array)
    overall_time_array = overall_time_array
    # time_array = time_array + 50e-3
    # s_input.plot_fft(fs = 10e3, max_x=500*hertz,min_x=0.1*hertz)

    defaultclock.dt = parameters['here_clock']

    set_device('cpp_standalone', directory= os.path.join(PRECOMPILED, code_name, str(index)))
    print('creating neurons')
    Fake_Neuron, Oscillator_neuron, TDE, HP0, HP1, HP2, nWTA, gWTA, Poisson_to_nWTA\
        = create_PLL(len(parameters['frequencies']),parameters['neurons_n'], overall_neuron_array,overall_time_array)
    for input_idx in range(len(parameters['frequencies'])):
        for neuron_idx in range(parameters['neurons_n']):
            idx = neuron_idx + parameters['neurons_n'] * input_idx
            Oscillator_neuron.I[idx] = parameters['I_minimum_osc'] + parameters['I_step_osc'] * neuron_idx
    print('creating synapses')
    NMDA_h_plus,S_h_plus,FB_syn,HP0_syn,HP1_syn,HP2_syn,to_nWTA,\
    from_gWTA, to_gWTA,Poisson_syn,normalize_WTA = create_synapses(Fake_Neuron,
                                                                                  Oscillator_neuron, TDE, HP0, HP1,
                                                                                  HP2, nWTA, gWTA,Poisson_to_nWTA)
    # w = generate_NMDA_matrix(Fake_Neuron, TDE)
    print('creating monitors')
    Fake_spikes, Osc_spikes, nWTA_spikes, HP0_spikes, HP1_spikes, HP2_spikes, TDE_spikes, Gwta_spikes, TDE_spikes = \
        create_monitors(Fake_Neuron, Oscillator_neuron, nWTA, HP0, HP1, HP2, TDE, gWTA)

    current_time = overall_time_array.max()
    # print(Oscillator_neuron.I)
    print('starting simulation')
    run(current_time * second, report = 'stdout')
    print('converting simulation data')
    # s_osc = dummy_spikestructure(t=Osc_spikes.t, i=Osc_spikes.i)
    # s_fake = dummy_spikestructure(t=Fake_spikes.t, i=Fake_spikes.i)
    s_tde = dummy_spikestructure(t=TDE_spikes.t, i=TDE_spikes.i)
    # s_hp0 = dummy_spikestructure(t=HP0_spikes.t, i=HP0_spikes.i)
    # s_hp1 = dummy_spikestructure(t=HP1_spikes.t, i=HP1_spikes.i)
    # s_hp2 = None
    # s_wta = None
    s_hp2 = dummy_spikestructure(t=HP2_spikes.t, i=HP2_spikes.i)
    s_wta = dummy_spikestructure(t=nWTA_spikes.t, i = nWTA_spikes.i)
    s_in = dummy_spikestructure(t=overall_time_array,i = overall_neuron_array)
    output_dict = {'s_in' : s_in,
                   'frequencies': parameters['frequencies'],
                   's_osc_intrinsic': s_osc_intrinsic_freq,
                   's_hp2' : s_hp2,
                   's_wta': s_wta,
                   's_tde' : s_tde}
    device.reinit()
    device.activate(directory=PRECOMPILED + code_name + str(index))
    return output_dict

parameters = Simulation_parameters.import_parameters()

parameters['neurons_n'] = 20
parameters['save_plots'] = True #This option enables the saving of the plots in the IMAGE folder
parameters['frequencies'] = [i for i in linspace(50,200,50)] #The frequencies you want to simulate
parameters['I_minimum_osc'] = 0.012 * pA  # The minimum CCUR current fed to the Osc neuron
parameters['I_step_osc'] = 0.0015 * pA  # The variation of CCUR current fed to the Osc neuron
parameters['parallelized'] = False #This term enables the code parallelization over multiple cores
parameters['numbers_of_cores'] = 2 #This term tells how many cores the simulator is allowed to use
parameters = create_image_folder(IMAGE_PATH, code_name, parameters)

with open(os.path.join(DATA_PATH,'parameters.csv'), 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in parameters.items():
       writer.writerow([key, value])
for key, val in parameters.items():
    exec(key + '=val')

# 4 ==================== Run the simulation in parallelized or serial way ====================

viridis = mycm.get_cmap('Set1', 10)
newcolors = viridis(np.linspace(0, 1, parameters['neurons_n']*len(parameters['frequencies'])))

output_dict = run_the_script([0, parameters])
print('starting elaboration of data')
my_coll = []

neuron_list = [i for i in range(parameters['neurons_n'])]

x_offset = 0
stimuli = []
for i in range(len(parameters['frequencies'])):
    stimuli += [i for _ in range(parameters['neurons_n'])]

which_s = 's_tde'
output_dict[which_s].save_data(path = os.path.join(DATA_PATH,  code_name + '_' + which_s + '.csv'))
which_s = 's_wta'
output_dict[which_s].save_data(path = os.path.join(DATA_PATH,  code_name + '_' + which_s + '.csv'))
which_s = 's_hp2'
output_dict[which_s].save_data(path = os.path.join(DATA_PATH, code_name + '_' +  which_s + '.csv'))
which_s = 's_in'
output_dict[which_s].save_data(path = os.path.join(DATA_PATH,  code_name + '_' + which_s + '.csv'))
print('saved')
