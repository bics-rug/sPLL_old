from utils.pytorch_models import *
import numpy as np
import matplotlib.pyplot as plt
def generate_frequencies_events(parameters):
    time_array = []
    neuron_array = []
    train_test = []
    trial_array = []
    labels = torch.zeros([3,len(parameters['frequencies'])*(parameters['trials_per_stimulus_test'] + parameters['trials_per_stimulus_train'])], device = parameters['device'])*torch.nan
    idx_to_write = 0
    print('creating frequencies...')
    for fr_ix,freq in enumerate(parameters['frequencies']):
        # print('creating frequency:',freq)
        time_array_temp = torch.linspace(0, parameters['tot_time']/parameters['clock_sim']*freq, np.floor(parameters['tot_time'] * freq).astype(int)) * parameters['clock_sim'] / freq
        if len(time_array_temp) == 0:
            time_array_temp = torch.array([0])
        standard_array = torch.ones(np.floor(parameters['tot_time'] * freq).astype(int))
        for trial in range(parameters['trials_per_stimulus_train']):
            if (parameters['jitter_var'] > 0) & (freq > 0):
                noise = torch.normal(parameters['clock_sim'] / (1 * freq),
                                         parameters['clock_sim'] / (parameters['jitter_var'] * freq),
                                     (len(time_array_temp),))
                # plt.plot(noise)
                # plt.show()
                # print('applied jitter = ',1 / (parameters['jitter_var']*freq))
            else:
                noise = torch.zeros([len(time_array_temp)])
            time_array_temp = time_array_temp + noise

            time_array_temp = time_array_temp[time_array_temp>0]
            plt.plot(time_array_temp)
            standard_array = torch.ones([len(time_array_temp)])
            time_array = np.append(time_array,time_array_temp)
            neuron_array = np.append(neuron_array,standard_array*(fr_ix))
            trial_array = np.append(trial_array,standard_array*(trial))
            train_test = np.append(train_test,standard_array*0)
            labels[0,idx_to_write] = freq
            labels[1,idx_to_write] = 0
            labels[2,idx_to_write] = fr_ix
            idx_to_write += 1


        for trial in range(parameters['trials_per_stimulus_test']):
            # print(idx_to_write)

            if (parameters['jitter_var'] > 0) & (freq > 0):
                noise = np.random.normal(-parameters['clock_sim'] / (1 * freq),
                                         parameters['clock_sim'] / (parameters['jitter_var'] * freq),
                                         len(time_array_temp))
            else:
                noise = np.zeros([len(time_array_temp)])
            # print(noise)
            time_array_temp = time_array_temp + noise
            time_array_temp = np.append(time_array_temp[time_array_temp>0],[0])
            standard_array = np.ones([len(time_array_temp)])
            time_array = np.append(time_array,time_array_temp)
            neuron_array = np.append(neuron_array,standard_array*(fr_ix))
            trial_array = np.append(trial_array,standard_array*(trial))
            train_test = np.append(train_test,standard_array*1)
            labels[0,idx_to_write] = freq
            labels[1,idx_to_write] = 1
            labels[2,idx_to_write] = fr_ix
            idx_to_write += 1

            # trial += 1
    plt.show()
    return time_array,neuron_array,train_test,labels, trial_array
def add_noise_to_spike(time_array,neuron_array,train_test,trial_array,parameters):

        # np.random.seed(parameters['seed'])

        print('adding noise to the signal')
        noise = np.random.normal(-parameters['clock_sim'] / (parameters['jitter_const']), parameters['clock_sim'] / (parameters['jitter_const']), len(time_array))
        time_array_noisy_neg = time_array + noise
        time_array_noisy = time_array_noisy_neg[time_array_noisy_neg > 0]
        neuron_array_noisy = neuron_array[time_array_noisy_neg > 0]
        train_test_noisy = train_test[time_array_noisy_neg > 0]
        trial_array_noisy = trial_array[time_array_noisy_neg > 0]
        return time_array_noisy,neuron_array_noisy,train_test_noisy,trial_array_noisy
def events_to_bins(time_array,neuron_array,train_test,labels,trial_array,parameters):
    dim0 = np.round(time_array/parameters['clock_sim']).astype(int)
    dim0_train = dim0[train_test == 0]
    dim0_test = dim0[train_test == 1]
    dim1 = neuron_array.astype(int)
    dim1_train = dim1[train_test == 0]
    dim1_test = dim1[train_test == 1]
    dim2 = trial_array.astype(int)
    dim2_train = dim2[train_test == 0]
    dim2_test = dim2[train_test == 1]
    # binned_spikes_train = torch.zeros([parameters['tot_time']/parameters['clock_sim'],len(parameters['frequencies'])*parameters['trail_per_stimulus_train']])
    # binned_spikes_test = torch.zeros([parameters['tot_time']/parameters['clock_sim'],len(parameters['frequencies'])*parameters['trail_per_stimulus_test']])

    binned_spikes_train = torch.sparse_coo_tensor(indices = torch.tensor(np.array([dim1_train,dim2_train,dim0_train])),values = [1 for i in range(len(dim1_train))],dtype = torch.int8, device=parameters['device'])
    binned_spikes_test = torch.sparse_coo_tensor(indices = torch.tensor(np.array([dim1_test,dim2_test,dim0_test])),values = [1 for i in range(len(dim1_test))],dtype = torch.int8, device=parameters['device'])
    # binned_spikes_train = binned_spikes_train[np.where(labels[1,:] == 0)]
    # binned_spikes_test = binned_spikes_test[np.where(labels[1,:] == 1)]
    return binned_spikes_train,binned_spikes_test
def input_freq_gen_events(parameters):
    # parameters['jitter_var'] = 0
    time_array,neuron_array,train_test,labels,trial_array_noisy = generate_frequencies_events(parameters)
    # if parameters['jitter_const'] > 0:
    #     time_array, neuron_array,train_test,trial_array_noisy = add_noise_to_spike(time_array,neuron_array,train_test,trial_array_noisy,parameters)
    # plt.scatter(time_array,neuron_array)
    freq = 48
    for trial in range(200):
        which_trial = trial_array_noisy == trial
        which_freq = neuron_array == freq
        is_test_tran = train_test == 0
        selected_times = time_array[which_freq & which_trial & is_test_tran]
        # print(len(selected_times))
        plt.plot(selected_times, [trial for i in range(len(selected_times))], '.')
    plt.show()
    spikes = events_to_bins(time_array, neuron_array,train_test,labels,trial_array_noisy,parameters)
    return spikes,labels

def spike_counter(data,dim = 0):
    count = torch.sparse.sum(data.type(torch.int64), dim = dim).to_dense()
    return count
def input_freq_gen_binned(freq_range,parameters):
    parameters['input_n'] = len(freq_range)
    in_spikes = torch.zeros((parameters['input_n'], parameters['neurons_n'],parameters['tot_time']), device=device)
    for i in range(parameters['input_n']):
        step = 1/(freq_range[i]*parameters['clock_sim'])
        in_spikes[i,:,int(1e-3/parameters['clock_sim'])::int(step)] = 1
    return in_spikes

def isi_calculator(data,parameters, return_d = False):
    if len(data.shape) == 3:
        tdx,idx_stim,idx = np.where(data.detach().cpu().numpy())
    else:
        idx, tdx = np.where(data.detach().cpu().numpy())

    mean_isi = np.zeros([parameters['input_n'],parameters['neurons_n']])
    for j in range(parameters['input_n']):
        for i in range(parameters['neurons_n']):
            diff = np.diff(tdx[(idx_stim == j) & (idx == i)])
            mean_isi[j,i] = np.mean(diff)
    import matplotlib.pyplot as plt
    plt.imshow(mean_isi.T)
    plt.show()
    if return_d:
        return mean_isi
    else:
        plt.plot(mean_isi,marker = '.')

def add_a_dim(mytensor,parameters):
    output_tensor = torch.zeros(mytensor.shape + tuple([1]) , device = parameters['device'])
    output_tensor[:,:,:,0] = mytensor
    return output_tensor

def reshape_my_tensor(mytensor):
    #we want to move from this: count dimensions are [trial , frequencies , neurons]
    # to [frequencies , trial , neurons]
    mytensor = torch.movedim(mytensor,1,0)
    #then me move to [frequenciesxtrial, neurons]
    return mytensor.reshape(mytensor.shape[0] * mytensor.shape[1], mytensor.shape[2])

def derivate_my_count(mytensor):
    deriv = torch.zeros_like(mytensor)[:,:,:-1]
    for trial in range(mytensor.shape[0]):
        for column in range(mytensor.shape[1] - 1):
            deriv[trial,column, :] = torch.diff(mytensor[trial,column, :])
        # mymax[trial,:] = np.argmax(deriv[trial,:,:], axis=0)
    return deriv