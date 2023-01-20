from brian2 import us, mV, ms, pA, pF, Quantity
import os

## override brian units -> make dimensionless
# us = 1.
# mV = 1.
# ms = 1.
# pA = 1.
# pF = 1.

Current_PATH = os.getcwd()
FILE_PATH = os.path.dirname(os.getcwd())
CODE_PATH = os.path.join(FILE_PATH, 'Code')


def import_parameters(search=None):
    parameters = {}

    ####### Brian2 simulation parameters #######
    
    parameters['noise'] = False
    parameters['save_plots'] = False
    parameters['Images_Folder'] = []
    parameters['frequencies'] = [0]
    parameters['input_poissonian'] = False

    # parameters['dt'] = 0.01 * ms
    parameters['clock_sim'] = 10 * us               # clock at which the simulation run
    parameters['neurons_n'] = 5                     # number of lines of PLLs
    parameters['input_n'] = 50

    parameters['sigma_v'] = 0 * mV                  # sigma of the noise to be added to the neurons
    parameters['tau_noise'] = 100 * ms              # period of the noise

    parameters['C_TDE'] = 0.01*pF
    parameters['tau_TDE'] = 10 * ms                 # time constant of the TDE's neuron
    parameters['tau_TDE_Ie_h'] = 10 * ms            # time constant of the TDE's facilitatory
    parameters['tau_TDE_Ie_NMDA_h'] = 1 * ms        # time constant of the TDE's trigger
    parameters['v_TDE_threshold'] = 20 * mV         # voltage threshold of the TDE's neuron
    parameters['refrac_TDE'] = 0.0001 * ms          # refractory period of the TDE's neuron
    parameters['qe_NMDA'] = 0.1                     # gain of the TDE's trigger
    parameters['Ie_th'] = 5 * pA                    # threshold of the TDE's trugger
    parameters['qe'] = 10 * pA                      # gain of the TDE's facilitatory
    parameters['reset_TDE'] = 0
    parameters['capped_TDE'] = True
    parameters['tau_TDE_Inoise'] = 1 * ms

    parameters['C_Osc'] = 0.01*pF
    parameters['tau_Osc'] = 10 * ms                 # time constant of Osc neuron
    parameters['tau_Osc_Ie'] = 0.1 * ms             # time constant of Osc excitatory synapse
    parameters['tau_Osc_Ii'] = 0.1 * ms             # time constant of Osc inhibitory synapse
    parameters['v_Osc_threshold'] = 10 * mV         # threshold of Osc neuron
    parameters['refrac_Osc'] = 0.001 * ms           # refractory period of Osc neuron
    parameters['I_minimum_osc'] = 0.012 * pA        # minimum CCUR current fed to Osc neuron
    parameters['I_step_osc'] = 0.002*pA             # variation of CCUR current fed to Osc neuron
    parameters['reset_osc'] = 0
    # parameters['I_minimum_osc'] = 1 * pA

    parameters['C_HP'] = 0.01*pF
    parameters['tau_HP'] = 1 * ms                  # time constant of the HP neuron
    parameters['tau_HP_Ie'] = 0.1 * ms             # time constant of the HP excitatory synapse
    parameters['tau_HP_Ii'] = 0.1 * ms             # time constant of the HP inhibitory synapse
    parameters['v_HP_threshold'] = 10 * mV         # threshold of the HP neuron
    parameters['refrac_HP'] = 0.1 * ms             # refractory period of the HP neuron

    parameters['tau_WTA'] = 1 * ms                 # time constant of the WTA neuron
    parameters['tau_WTA_Ie'] = 10 * ms             # time constant of the WTA excitatory synapse
    parameters['tau_WTA_Ii'] = 10 * ms             # time constant of the WTA inhibitory synapse
    parameters['v_WTA_threshold'] = 10 * mV        # threshold of the WTA inhibitory synapse
    parameters['refrac_WTA'] = 0.001 * ms          # refractory period of the HP neuron
    parameters['WTA_stable_current'] = 0* pA       # DC input current fed to the i-WTA

    parameters['TDE_to_Osc_current'] = 0.1 * pA    # current from TDE to the OSC
    parameters['TDE_to_HP0_current'] = 1 * pA      # current from TDE to the first stage of HP
    parameters['HP0_to_HP1_current'] = 1 * pA      # current from 1st HP to the second stage
    parameters['HP1_to_HP2_current'] = 1 * pA      # current from 2nd HP to the third stage
    parameters['HP2_to_nWTA_current'] = 1 * pA     # current from 3rd HP to WTA
    parameters['HP2_to_nWTA_leftc'] = 0.05 * pA    # current from 3rd HP to WTA on the left
    parameters['HP2_to_nWTA_rightc'] = 0.05 * pA   # current from 3rd HP to WTA on the right
    parameters['nWTA_to_gWTA_current'] = 1 * pA    # current from WTA to Global Inh mechanism
    parameters['gWTA_to_nWTA_current'] = 1 * pA    # current from Global Inh mechanism to WTA
    parameters['Fake_to_nWTA_current'] = 0 * pA    # current from Input to the WTA
    parameters['Fake_to_gWTA_current'] = 0 * pA    # current from Input to Global Inh mechanism
    parameters['in_to_TDE_current'] = 0 * pA

    parameters['trials_per_stimulus_train'] = 1
    parameters['trials_per_stimulus_test'] = 0

    ####### skin simulation parameters #######

    parameters['analog_clock'] = 0.01 * ms          # clock of the skin simulator

    ## texture parameters
    parameters['spatial_frequency'] = [5, 10, 15]
    parameters['spatial_frequency_here'] = []
    parameters['orientations'] = [90 for i in range(len(parameters['spatial_frequency']))]
    parameters['lengths'] = [300 for i in range(len(parameters['spatial_frequency']))]
    parameters['widths'] = parameters['spatial_frequency']
    parameters['depths'] = [1 for i in range(len(parameters['spatial_frequency']))]
    parameters['numbers'] = [20 for i in range(len(parameters['spatial_frequency']))]
    parameters['velocities'] = [100 for i in range(len(parameters['spatial_frequency']))]

    parameters['Data_PATH'] = os.makedirs(os.path.join(FILE_PATH, 'Data'), exist_ok=True)
    parameters['PRECOMPILED'] = os.makedirs(os.path.join(FILE_PATH, 'Precompiled'), exist_ok=True)
    parameters['IMAGE_PATH'] = os.makedirs(os.path.join(FILE_PATH, 'Images'), exist_ok=True)
    parameters['code_name'] = 'PLL_network'

    if search != None:
        parameters_searched = {search : parameters[search]}
        return parameters_searched
    else:
        return parameters


def import_parameters_dimensionless(search = None):
    dimless_params = {}
    parameters = import_parameters(search)
    for key in parameters.keys():
        if type(parameters[key]) == Quantity:
            dimless_params[key] = float(parameters[key])
        else:
            dimless_params[key] = parameters[key]
    return dimless_params
