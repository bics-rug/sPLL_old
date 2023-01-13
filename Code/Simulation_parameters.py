from brian2 import us,mV,ms,pA,pF,Quantity,get_dimensions
from utils.utils_misc import *
import os, sys
Current_PATH = os.getcwd()
FILE_PATH = os.path.dirname(os.getcwd())
CODE_PATH = os.path.join(FILE_PATH, 'Code')
def import_parameters(search = None):

    parameters = {}

    ####### Brian2 simulation parameters #######
    parameters['noise'] = False
    parameters['save_plots'] = False
    parameters['Images_Folder'] = []
    parameters['frequencies'] = [0]
    parameters['input_poissonian'] = False

    # parameters['dt'] = 0.01 * ms
    parameters['clock_sim'] = 10 * us             #The clock at which the simulation run
    parameters['neurons_n'] = 5          #The number of lines of PLLs
    parameters['input_n'] = 50

    parameters['sigma_v'] = 0 * mV                 #The sigma of the noise to be added to the neurons
    parameters['tau_noise'] = 100 * ms             #The period of the noise

    parameters['C_TDE'] = 0.01*pF
    parameters['tau_TDE'] = 10 * ms                #The time constant of the TDE's neuron
    parameters['tau_TDE_Ie_h'] = 10 * ms           #The time constant of the TDE's facilitatory
    parameters['tau_TDE_Ie_NMDA_h'] = 1 * ms       #The time constant of the TDE's trigger
    parameters['v_TDE_threshold'] = 20 * mV        #The voltage threshold of the TDE's neuron
    parameters['refrac_TDE'] = 0.0001 * ms           #The refractory period of the TDE's neuron
    parameters['qe_NMDA'] = 0.1                    #The gain of the TDE's trigger
    parameters['Ie_th'] = 5 * pA                   #The threshold of the TDE's trugger
    parameters['qe'] = 10 * pA                     #The gain of the TDE's facilitatory
    parameters['reset_TDE'] = 0
    parameters['capped_TDE'] = True
    parameters['tau_TDE_Inoise'] = 1 * ms

    parameters['C_Osc'] = 0.01*pF
    parameters['tau_Osc'] = 10 * ms                #The time constant of the Osc neuron
    parameters['tau_Osc_Ie'] = 0.1 * ms            #The time constant of the Osc excitatory synapse
    parameters['tau_Osc_Ii'] = 0.1 * ms            #The time constant of the Osc inhibitory synapse
    parameters['v_Osc_threshold'] = 10 * mV        #The threshold of the Osc neuron
    parameters['refrac_Osc'] = 0.001 * ms              #The refractory period of the Osc neuron
    parameters['I_minimum_osc'] = 0.012 * pA  # The minimum CCUR current fed to the Osc neuron
    # parameters['I_minimum_osc'] = 1 * pA
    parameters['I_step_osc'] = 0.002*pA           #The variation of CCUR current fed to the Osc neuron
    parameters['reset_osc'] = 0

    parameters['C_HP'] = 0.01*pF
    parameters['tau_HP'] = 1 * ms                  #The time constant of the HP neuron
    parameters['tau_HP_Ie'] = 0.1 * ms             #The time constant of the HP excitatory synapse
    parameters['tau_HP_Ii'] = 0.1 * ms             #The time constant of the HP inhibitory synapse
    parameters['v_HP_threshold'] = 10 * mV         #The threshold of the HP neuron
    parameters['refrac_HP'] = 0.1 * ms             #The refractory period of the HP neuron

    parameters['tau_WTA'] = 1 * ms                 #The time constant of the WTA neuron
    parameters['tau_WTA_Ie'] = 10 * ms             #The time constant of the WTA excitatory synapse
    parameters['tau_WTA_Ii'] = 10 * ms             #The time constant of the WTA inhibitory synapse
    parameters['v_WTA_threshold'] = 10 * mV        #The threshold of the WTA inhibitory synapse
    parameters['refrac_WTA'] = 0.001 * ms          #The refractory period of the HP neuron
    parameters['WTA_stable_current'] = 0* pA       #The DC input current fed to the i-WTA

    parameters['TDE_to_Osc_current'] = 0.1 * pA    #The current from the TDE to the OSC
    parameters['TDE_to_HP0_current'] = 1 * pA      #The current from the TDE to the first stage of HP
    parameters['HP0_to_HP1_current'] = 1 * pA      #The current from the first stage of HP to the second stage
    parameters['HP1_to_HP2_current'] = 1 * pA      #The current from the second stage of HP to the third stage
    parameters['HP2_to_nWTA_current'] = 1 * pA     #The current from the third stage of HP to the WTA
    parameters['HP2_to_nWTA_leftc'] = 0.05 * pA    #The current from the third stage of HP to the WTA on the left
    parameters['HP2_to_nWTA_rightc'] = 0.05 * pA   #The current from the third stage of HP to the WTA on the right
    parameters['nWTA_to_gWTA_current'] = 1 * pA    #The current from the WTA to the Global Inh mechanism
    parameters['gWTA_to_nWTA_current'] = 1 * pA    #The current from the Global Inh mechanism to the WTA
    parameters['Fake_to_nWTA_current'] = 0 * pA    #The current from the Input to the WTA
    parameters['Fake_to_gWTA_current'] = 0 * pA    #The current from the Input to the Global Inh mechanism
    parameters['in_to_TDE_current'] = 0 * pA

    parameters['trials_per_stimulus_train'] = 1
    parameters['trials_per_stimulus_test'] = 0
    ####### Skin simulation parameters #######

    parameters['analog_clock'] = 0.01 * ms          #The clock of the skin simulator

    parameters['spatial_frequency'] = [5, 10, 15]  #The spatial freq of the texture
    parameters['spatial_frequency_here'] = []
    parameters['orientations'] = [90 for i in range(len(parameters['spatial_frequency']))] #The orientations of the texture
    parameters['lengths'] = [300 for i in range(len(parameters['spatial_frequency']))] #The lengths of the texture
    parameters['widths'] = parameters['spatial_frequency'] #The widths of the texture
    spaces = parameters['spatial_frequency'] #The spaces of the texture
    parameters['depths'] = [1 for i in range(len(parameters['spatial_frequency']))] #The depths of the texture
    parameters['numbers'] = [20 for i in range(len(parameters['spatial_frequency']))] #The numbers of the texture
    parameters['velocities'] = [100 for i in range(len(parameters['spatial_frequency']))] #The sliding velocities

    parameters['Data_PATH'] = create_folder(os.path.join(FILE_PATH, 'Data'))
    parameters['PRECOMPILED'] = create_folder(os.path.join(FILE_PATH, 'Precompiled'))
    parameters['IMAGE_PATH'] = create_folder(os.path.join(FILE_PATH, 'Images'))
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