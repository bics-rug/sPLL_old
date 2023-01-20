import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from sPLL_models import sPLL
from Simulation_parameters import import_parameters_dimensionless


if torch.cuda.is_available():
    print("Single GPU detected. Setting up the simulation there.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(torch.device('cuda:0'))
    torch.cuda.empty_cache()
else:
    print("No GPU detected. Running on CPU.")
    device = torch.device("cpu")


sim_time = 0.1
clock_sim = 0.01e-3
trial_n = 100
freqs = [(i)*50+20 for i in range(10)]
noise = 1e-2

parameters = import_parameters_dimensionless()

print(parameters)
input()

no_grad = [
    'tau_TDE',              # time constant of TDE's neuron
    'tau_TDE_Ie_h',         # time constant of TDE's facilitatory
    'tau_TDE_Ie_NMDA_h',    # time constant of TDE's trigger
    'v_TDE_threshold',      # voltage threshold of TDE's neuron
    'refrac_TDE',           # refractory period of TDE's neuron
    'qe_NMDA',              # gain of TDE's trigger
    'C_Osc',
    'tau_Osc',              # time constant of Osc neuron
    'tau_Osc_Ie',           # time constant of Osc excitatory synapse
    'tau_Osc_Ii',           # time constant of Osc inhibitory synapse
]
yes_grad = [
    'Ie_th',                # threshold of TDE's trigger
    'qe',                   # gain of TDE's facilitatory
    'TDE_to_Osc_current',   # current from TDE to Osc
    'TDE_to_HP0_current',   # current from TDE to first HP
]

for param in no_grad:
    # if refrac_TDE then add `.to(device)`
    parameters[param] = nn.Parameter(torch.Tensor([parameters[param]]), requires_grad=False)

for param in yes_grad:
    parameters[param] = nn.Parameter(torch.Tensor([parameters[param]]), requires_grad=True)

h1 = nn.Parameter(torch.Tensor([-1e-12,-1e-12,-1e-12]),requires_grad=False)

parameters['trials_per_stimulus'] = trial_n
parameters['frequencies'] = freqs

parameters['neurons_n'] = len(freqs)
parameters['frequencies'] = freqs
parameters['clock_sim'] = clock_sim
parameters['device'] = device
parameters['tot_time'] = sim_time

def generate_one_freq(f_ix, freq):
    trial_vec = torch.linspace(0, trial_n - 1, trial_n)
    trials = torch.ones([np.floor(sim_time * freq).astype(int), trial_n])

    time_array_temp = torch.linspace(0, sim_time / clock_sim * freq,
                                     np.floor(sim_time * freq).astype(int)) * clock_sim / freq
    time_array = trials.T * time_array_temp
    check = (torch.randn(time_array.shape) * noise / freq) + noise / freq
    time_array_noisy = time_array + check
    # plt.eventplot(time_array_noisy)
    # plt.show()
    freq_array = torch.ones(time_array.shape) * f_ix
    trial_array = (torch.ones(time_array.shape).T * trial_vec).T
    labels = torch.ones(trial_n) * f_ix
    time_array_noisy = time_array_noisy.flatten()
    freq_array = freq_array.flatten()
    trial_array = trial_array.flatten()
    return time_array_noisy, freq_array, trial_array, labels

time_array_noisy_list = []
freq_array_list = []
trial_array_list = []
labels_list = []
for f_ix, freq in enumerate(freqs):
    time_array_noisy, freq_array, trial_array, labels = generate_one_freq(f_ix,freq)
    time_array_noisy_list.append(time_array_noisy)
    freq_array_list.append(freq_array)
    trial_array_list.append(trial_array)
    labels_list.append(labels)

time_array_noisy_tensor = (torch.cat(time_array_noisy_list)/clock_sim).type(torch.int)
freq_array_tensor = torch.cat(freq_array_list).type(torch.int)
trial_array_tensor = torch.cat(trial_array_list).type(torch.int)
labels_tensor = torch.cat(labels_list)

pos_values = time_array_noisy_tensor>=0
time_array_noisy_tensor = time_array_noisy_tensor[pos_values]
trial_array_tensor = trial_array_tensor[pos_values]
freq_array_tensor = freq_array_tensor[pos_values]

cumulative_tensor = torch.sparse_coo_tensor(indices = torch.tensor(np.array([np.array(time_array_noisy_tensor),np.array(freq_array_tensor),np.array(trial_array_tensor)])),values=torch.ones_like(time_array_noisy_tensor))
cumulative_tensor = cumulative_tensor.to_dense().flatten(start_dim=1, end_dim=2)
x_train, x_test, y_train, y_test = train_test_split(
    cumulative_tensor.T, labels_tensor, test_size=0.2, shuffle=True, stratify=labels_tensor)
ds_train = TensorDataset(x_train,y_train)
ds_test = TensorDataset(x_test,y_test)

N_WORKERS = 0  # before: 4
dl_train = DataLoader(ds_train, batch_size=5, shuffle=True, num_workers=N_WORKERS, pin_memory=True)
dl_test = DataLoader(ds_test, batch_size=5, shuffle=True, num_workers=N_WORKERS, pin_memory=True)

softmin_fn = nn.Softmin(dim=1)
# loss_fn = nn.NLLLoss()
loss_fn = nn.CrossEntropyLoss()
torch.autograd.set_detect_anomaly(True)
currents = []
loss = []
acc = []


def objective(config):
    sPLL_array = sPLL(config, device).to(device)
    # L2_array = LIF_neuron_l2(config).to(device)
    optimizer = torch.optim.Adamax(sPLL_array.parameters(), lr=1e-16, betas=(0.9, 0.995))

    for i in range(5):
        parameters_tmp = {'trials_per_stimulus':0}
        acc_list = []
        for x_local,y_local in dl_train:
            spk_tde = []
            spk_lif = []
            spk_l2 = []
            vmem_tde = []
            x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(
                device, non_blocking=True
            )

            parameters_tmp['trials_per_stimulus'] = x_local.shape[0]
            sPLL_array.initialize_state(parameters=parameters_tmp)
            # L2_array.initialize_state(parameters = parameters_tmp)
            for t in range(0, x_local.shape[1]):
                sPLL_array(x_local[:,t])
                spk_l2.append(sPLL_array.spikes_TDE)
                spk_lif.append(sPLL_array.spikes_LIF)
                spk_tde.append(sPLL_array.spikes_TDE)

                # print(self.spike_record_LIF)
            spk_tde = torch.stack(spk_tde)

            # spk_l2 = torch.stack(spk_l2)
            spk_tde_sum = torch.sum(spk_tde,dim = 0)
            total_spikes = torch.sum(spk_tde)
            tmp = torch.ones([spk_tde_sum.shape[0], 1]).to(device)
            tmp = torch.mul(tmp, spk_tde_sum[:, 0])[0]
            spk_tde_sum = torch.vstack([tmp, spk_tde_sum.T]).T

            spk_tde_diff = torch.diff(spk_tde_sum, dim=1)
            # print('spk_tde_diff before',spk_tde_diff)

            # spk_tde_diff[spk_tde_diff > 0] = 0
            # spk_tde_diff = torch.abs(spk_tde_diff)
            # print('spk_tde_sum',spk_tde_sum)
            # print('spk_tde_diff',spk_tde_diff)
            loss_val = loss_fn(softmin_fn(spk_tde_diff), y_local.type(torch.int64))

            # p_y = log_softmax_fn(spk_tde_sum)
            # loss_val = loss_fn(softmax_fn(spk_l2_sum), y_local.type(torch.int64))
            # loss_val += 10000 * (total_spikes == 0)
            # loss_val += total_spikes

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            with torch.no_grad():
                mymin = torch.argmin(spk_tde_diff.clone().detach().cpu(), dim=1)

                plt.plot([freqs[int(j)] for j in y_local.clone().detach().cpu().numpy()], mymin, '.')
                plt.show()
                # compare to labels
                tmp = np.mean((y_local == mymin.to(device)).detach().cpu().numpy())
                acc_list.append(tmp)
                print('loss',loss_val,'acc', tmp,'winner',mymin,'label',y_local)

        input_events = torch.where(x_local[0, :])
        plt.scatter(input_events[0].clone().detach().cpu(), torch.zeros_like(input_events[0]).clone().detach().cpu())
        spk_lif = torch.stack(spk_lif)
        spk_lif_events = torch.where(spk_lif.clone().detach()[:, 0, :])
        # spk_l2_events = torch.where(spk_l2.clone().detach()[:,0,:])
        plt.scatter(spk_lif_events[0].clone().detach().cpu(), spk_lif_events[1].clone().detach().cpu() + 1)
        spk_tde_events = torch.where(spk_tde.clone().detach()[:, 0, :])
        plt.scatter(spk_tde_events[0].clone().detach().cpu(), spk_tde_events[1].clone().detach().cpu() + parameters['neurons_n'] + 1)
        # plt.scatter(spk_l2_events[0], spk_l2_events[1] + 2*parameters['neurons_n'] + 1)
        plt.figure()
        plt.yticks([i for i in range(len(y_local.clone().detach().cpu().numpy()))],
                   [freqs[int(j)] for j in y_local.clone().detach().cpu().numpy()])
        plt.imshow(spk_tde_diff.detach().cpu(), aspect='auto')
        plt.figure()
        plt.yticks([i for i in range(len(y_local.clone().detach().cpu().numpy()))],
                   [freqs[int(j)] for j in y_local.clone().detach().cpu().numpy()])
        plt.imshow(spk_tde_sum.detach().cpu(), aspect='auto')
        # plt.show()
        plt.figure()
        currents.append(optimizer.param_groups[0]['params'][0].clone().detach().cpu().numpy())
        loss.append(loss_val.clone().detach().cpu().numpy())
        currents_to_plot = np.stack(currents)
        # print(currents_to_plot)
        plt.plot(currents_to_plot)
        plt.title('Currents')
        plt.show()
        plt.title('Loss')
        plt.plot(loss)
        plt.figure()
        plt.title('Acc')
        plt.plot(np.mean(acc_list))
        acc.append(tmp)
        plt.show()

objective(config = parameters)
print('ciao')
