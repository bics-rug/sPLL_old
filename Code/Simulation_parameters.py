from brian2 import us,mV,ms,pA
def import_parameters(search = None):
    parameters = {}

    ####### Brian2 simulation parameters #######
    parameters['save_plots'] = False
    parameters['Images_Folder'] = []
    parameters['frequencies'] = []

    parameters['here_clock'] = 10 * us             #The clock at which the simulation run
    parameters['neurons_n'] = 200                   #The number of lines of PLLs

    parameters['sigma_v'] = 0 * mV                 #The sigma of the noise to be added to the neurons
    parameters['tau_noise'] = 100 * ms             #The period of the noise

    parameters['tau_TDE'] = 10 * ms                #The time constant of the TDE's neuron
    parameters['tau_TDE_Ie_h'] = 10 * ms           #The time constant of the TDE's facilitatory
    parameters['tau_TDE_Ie_NMDA_h'] = 1 * ms       #The time constant of the TDE's trigger
    parameters['v_TDE_threshold'] = 20 * mV        #The voltage threshold of the TDE's neuron
    parameters['refrac_TDE'] = 0.01 * ms           #The refractory period of the TDE's neuron
    parameters['qe_NMDA'] = 0.1                    #The gain of the TDE's trigger
    parameters['Ie_th'] = 5 * pA                   #The threshold of the TDE's trugger
    parameters['qe'] = 10 * pA                     #The gain of the TDE's facilitatory

    parameters['tau_Osc'] = 10 * ms                #The time constant of the Osc neuron
    parameters['tau_Osc_Ie'] = 0.1 * ms            #The time constant of the Osc excitatory synapse
    parameters['tau_Osc_Ii'] = 0.1 * ms            #The time constant of the Osc inhibitory synapse
    parameters['v_Osc_threshold'] = 10 * mV        #The threshold of the Osc neuron
    parameters['refrac_Osc'] = 0.1 * ms              #The refractory period of the Osc neuron
    parameters['I_minimum_osc'] = 0.015*pA         #The minimum CCUR current fed to the Osc neuron
    parameters['I_step_osc'] = 0.0015*pA           #The variation of CCUR current fed to the Osc neuron

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
    parameters['TDE_to_HP0_current'] = 2 * pA      #The current from the TDE to the first stage of HP
    parameters['HP0_to_HP1_current'] = 2 * pA      #The current from the first stage of HP to the second stage
    parameters['HP1_to_HP2_current'] = 2 * pA      #The current from the second stage of HP to the third stage
    parameters['HP2_to_nWTA_current'] = 1 * pA     #The current from the third stage of HP to the WTA
    parameters['HP2_to_nWTA_leftc'] = 0.05 * pA    #The current from the third stage of HP to the WTA on the left
    parameters['HP2_to_nWTA_rightc'] = 0.05 * pA   #The current from the third stage of HP to the WTA on the right
    parameters['nWTA_to_gWTA_current'] = 1 * pA    #The current from the WTA to the Global Inh mechanism
    parameters['gWTA_to_nWTA_current'] = 100 * pA    #The current from the Global Inh mechanism to the WTA
    parameters['Fake_to_nWTA_current'] = 0 * pA    #The current from the Input to the WTA
    parameters['Fake_to_gWTA_current'] = 0 * pA    #The current from the Input to the Global Inh mechanism

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
    if search != None:
        parameters_searched = {search : parameters[search]}
        return parameters_searched
    else:
        return parameters
