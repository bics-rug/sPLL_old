import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# from sPLL_spytorch_lib import *
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from collections import namedtuple
from Simulation_parameters import import_parameters_dimensionless
# from ray import tune,air
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
if torch.cuda.is_available():
    print("Single GPU detected. Setting up the simulation there.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(torch.device('cuda:0'))
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("No GPU detected. Running on CPU.")
# from utils.utils_spikes_local import generate_spikes_from_freq


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 20.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


activation = SurrGradSpike.apply

class LIF_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'I','count_refr'])

    def __init__(self, parameters):
        super(LIF_neuron, self).__init__()
        self.C = parameters['C_Osc']
        self.thr = parameters['v_Osc_threshold']
        self.state = None
        self.bs = parameters['trials_per_stimulus']
        self.n = parameters['neurons_n']
        self.st_n = len(parameters['frequencies'])
        self.tau_syn = parameters['tau_Osc_Ie']
        self.tau_neu = parameters['tau_Osc']
        self.gain_syn = parameters['TDE_to_Osc_current']
        self.dt = parameters['clock_sim']
        self.Vr = parameters['reset_osc']
        self.device = parameters['device']
        self.I_minimum_osc = parameters['I_minimum_osc']
        self.I_step_osc = parameters['I_step_osc']
        self.refrac_Osc = parameters['refrac_Osc']
        self.steady_current = 0
        self.refr = self.refrac_Osc/self.dt

    def initialize_state(self,parameters):
        self.state = None
        self.bs = parameters['trials_per_stimulus']
    def forward(self, input):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones((self.bs,self.n), device=input.device),
                                          I=torch.zeros((self.bs,self.n), device=input.device),
                                          count_refr = torch.zeros((self.bs,self.n), device=input.device)
                                          )

        V = self.state.V
        I = self.state.I
        count_refr = self.state.count_refr

        I += -self.dt * I / self.tau_syn + self.gain_syn.item() * input

        V += self.dt * (-V / self.tau_neu + (I + self.steady_current) / self.C)

        spk = activation(V - self.thr)
        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)
        V = (1 - spk) * V * (count_refr <= 0) + spk * self.Vr
        # print(V.shape)
        # print(V)
        self.state = self.NeuronState(V=V, I=I,count_refr = count_refr)
        del V,I,count_refr
        return spk


class LIF_neuron_l2(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'I','count_refr'])

    def __init__(self, parameters):
        super(LIF_neuron_l2, self).__init__()
        self.C = parameters['C_l2']
        self.thr = parameters['v_l2_threshold']
        self.state = None
        self.bs = parameters['trials_per_stimulus']
        self.n = parameters['neurons_n']
        self.st_n = len(parameters['frequencies'])
        self.tau_syn = parameters['tau_l2_Ie']
        self.tau_neu = parameters['tau_l2']
        self.gain_syn = parameters['TDE_to_l2_current']
        self.dt = parameters['clock_sim']
        self.Vr = parameters['reset_l2']
        self.device = parameters['device']
        self.I_minimum_l2 = parameters['I_minimum_l2']
        self.I_step_l2 = parameters['I_step_l2']
        self.refrac_l2 = parameters['refrac_l2']
        self.steady_current = 1e-15
        self.refr = self.refrac_l2/self.dt

    def initialize_state(self,parameters):
        self.state = None
        self.bs = parameters['trials_per_stimulus']
    def forward(self, input_current):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones((self.bs,self.n), device=input_current.device),
                                          I=torch.zeros((self.bs,self.n), device=input_current.device),
                                          count_refr = torch.zeros((self.bs,self.n), device=input_current.device)
                                          )

        V = self.state.V
        I = self.state.I
        count_refr = self.state.count_refr

        # I += -self.dt * I / self.tau_syn + self.gain_syn.item() * input

        V += self.dt * (-V / self.tau_neu + (input_current+self.steady_current) / self.C)

        spk = activation(V - self.thr)
        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)
        V = (1 - spk) * V * (count_refr <= 0) + spk * self.Vr
        # print(V.shape)
        # print(V)
        self.state = self.NeuronState(V=V, I=I,count_refr = count_refr)
        del V,I,count_refr
        return spk


class TDE(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i_trg', 'i_fac','count_refr'])

    def __init__(self, parameters):
        super(TDE, self).__init__()
        self.C = parameters['C_TDE']
        self.thr = parameters['v_TDE_threshold']
        self.state = None
        self.n = parameters['neurons_n']
        self.st_n = len(parameters['frequencies'])
        self.tau_trg = parameters['tau_TDE_Ie_NMDA_h']
        self.tau_fac = parameters['tau_TDE_Ie_h']
        self.gain_fac = parameters['qe']
        self.gain_trg = parameters['qe_NMDA']
        self.ifac_thr = parameters['Ie_th']
        self.beta = parameters['tau_TDE']
        self.dt = parameters['clock_sim']
        self.Vr = parameters['reset_TDE']
        self.capped = parameters['capped_TDE']
        self.refr = parameters['refrac_TDE']/self.dt
        self.refr.to(device)

    def initialize_state(self,parameters):
        self.state = None
        self.bs = parameters['trials_per_stimulus']
    def forward(self, input_trg, input_fac):
        if self.state is None:
            self.state = self.NeuronState(V=torch.zeros(input_trg.shape, device=input_trg.device),
                                          i_trg=torch.zeros(input_trg.shape, device=input_trg.device),
                                          i_fac=torch.zeros(input_trg.shape, device=input_trg.device),count_refr = torch.zeros((self.bs,self.n), device=input_trg.device)
                                          )
        assert input_trg.shape == input_fac.shape, "TRG and FAC should have same dimensions, while: shape trg: " + str(input_trg.shape) + ", shape fac: " + str(input_fac.shape)
        v = self.state.V
        i_trg = self.state.i_trg
        count_refr = self.state.count_refr
        i_fac = input_fac * self.gain_fac + self.state.i_fac * (1-input_fac)
        i_fac += -self.dt / self.tau_fac * i_fac
        # i_fac = i_fac * ((i_fac < 10) | (self.capped == False)) + 10 * ((i_fac >= 10) & (self.capped == True))

        i_trg += -self.dt / self.tau_trg * i_trg
        i_trg += input_trg * self.gain_trg * i_fac * (i_fac > self.ifac_thr)
        #print('i_trg',i_trg,'i_fac',i_fac)
        v += self.dt * (- v / self.beta + i_trg / self.C)
        # print(v)
        spk = activation(v - self.thr)
        # print('count_refr',count_refr)
        # print('spk',spk)
        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)
        v = (1 - spk) * v * (count_refr <= 0) + spk * self.Vr

        self.state = self.NeuronState(V=v, i_trg=i_trg, i_fac=i_fac,count_refr= count_refr)
        return spk

class sPLL(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i_trg', 'i_fac'])

    def __init__(self, parameters):
        super(sPLL, self).__init__()
        self.TDE = TDE(parameters).to(device)
        self.LIF = LIF_neuron(parameters).to(device)
        self.st_n = len(parameters['frequencies'])

        self.n_out = parameters['neurons_n']
        self.n_in = 1
        self.tot_time = int(parameters['tot_time']/parameters['clock_sim'])
        self.device = parameters['device']

        # self.i_trg = torch.zeros((self.bs,self.n_out), device=self.device).to_sparse()
        # self.i_fac = torch.zeros((self.bs,self.n_out), device=self.device).to_sparse()
        # self.i_syn = torch.zeros((self.bs,self.n_out), device=self.device).to_sparse()
        self.spike_record_LIF = []
        self.spike_record_TDE = []
        self.tmp_current_list = nn.Parameter(torch.linspace(parameters['I_minimum_osc'],parameters['I_minimum_osc'] + parameters['I_step_osc']*self.n_out,self.n_out),requires_grad=True)
        print('ciao')
    def initialize_state(self,parameters):
        self.LIF.initialize_state(parameters)
        self.TDE.initialize_state(parameters)
        self.bs = parameters['trials_per_stimulus']
        self.input_fac = torch.zeros((self.bs,self.n_out), device=self.device)
        self.spikes_TDE_prev = torch.zeros((self.bs,self.n_out), device=self.device).to_sparse()
        self.spikes_TDE = torch.zeros((self.bs,self.n_out), device=self.device).to_sparse()
        self.spikes_LIF = torch.zeros((self.bs,self.n_out), device=self.device).to_sparse()
        self.current_list = self.tmp_current_list*torch.ones_like(self.input_fac)
        self.LIF.steady_current = self.current_list
    def forward(self, input):


                input_expanded = input*torch.ones([input.shape[0],self.n_out]).to(self.device).T
                self.spikes_LIF = self.LIF(self.spikes_TDE_prev)

                self.spikes_TDE = self.TDE(input_expanded.T, self.spikes_LIF)

                # self.i_trg = self.TDE.state.i_trg
                # self.i_fac = self.TDE.state.i_fac
                # self.i_syn = self.LIF.state.I
                self.spikes_TDE_prev = self.spikes_TDE.clone()

sim_time = 0.1
clock_sim = 0.01e-3
trial_n = 100
freqs = [(i)*50+20 for i in range(10)]
noise = 1e-2
parameters = import_parameters_dimensionless()
# parameters = {}
# parameters['C_TDE'] = tune.uniform(0.01e-12,1e-12)
parameters['tau_TDE'] = nn.Parameter(torch.Tensor([parameters['tau_TDE']]), requires_grad=False)
# parameters['tau_TDE'] = nn.Parameter(torch.Tensor([tune.uniform(10e-3,20-3)]), requires_grad=False)                #The time constant of the TDE's neuron#The time constant of the TDE's neuron
parameters['tau_TDE_Ie_h'] = nn.Parameter(torch.Tensor([parameters['tau_TDE_Ie_h']]), requires_grad=False)           #The time constant of the TDE's facilitatory
parameters['tau_TDE_Ie_NMDA_h'] = nn.Parameter(torch.Tensor([parameters['tau_TDE_Ie_NMDA_h']]), requires_grad=False)       #The time constant of the TDE's trigger
parameters['v_TDE_threshold'] = nn.Parameter(torch.Tensor([parameters['v_TDE_threshold']]), requires_grad=False)        #The voltage threshold of the TDE's neuron
parameters['refrac_TDE'] = nn.Parameter(torch.Tensor([parameters['refrac_TDE']]), requires_grad=False).to(device)        #The refractory period of the TDE's neuron
parameters['qe_NMDA'] = nn.Parameter(torch.Tensor([parameters['qe_NMDA']]), requires_grad=False)                    #The gain of the TDE's trigger
parameters['Ie_th'] = nn.Parameter(torch.Tensor([parameters['Ie_th']]), requires_grad=True)                   #The threshold of the TDE's trugger
parameters['qe'] = nn.Parameter(torch.Tensor([parameters['qe']]), requires_grad=True)                     #The gain of the TDE's facilitatory
# parameters['reset_TDE'] = 0
# parameters['capped_TDE'] = True
# parameters['tau_TDE_Inoise'] = 0
# parameters['clock_sim'] = clock_sim
parameters['trials_per_stimulus'] = trial_n
parameters['frequencies'] = freqs

parameters['C_Osc'] = nn.Parameter(torch.Tensor([parameters['C_Osc']]), requires_grad=False)
parameters['tau_Osc'] = nn.Parameter(torch.Tensor([parameters['tau_Osc']]), requires_grad=False)  # The time constant of the Osc neuron
parameters['tau_Osc_Ie'] = nn.Parameter(torch.Tensor([parameters['tau_Osc_Ie']]), requires_grad=False)  # The time constant of the Osc excitatory synapse
parameters['tau_Osc_Ii'] = nn.Parameter(torch.Tensor([parameters['tau_Osc_Ii']]), requires_grad=False)  # The time constant of the Osc inhibitory synapse
# parameters['v_Osc_threshold'] = 10e-3  # The threshold of the Osc neuron
# parameters['refrac_Osc'] = 0.1e-3  # The refractory period of the Osc neuron
# parameters['I_minimum_osc'] = 0.012e-12  # The minimum CCUR current fed to the Osc neuron
# # parameters['I_minimum_osc'] = 1 * pA
# parameters['I_step_osc'] = 0.0015e-12  # The variation of CCUR current fed to the Osc neuron
# parameters['reset_osc'] = 0

# parameters['C_l2'] = 0.01e-12
# parameters['tau_l2'] = 10e-3  # The time constant of the l2 neuron
# parameters['tau_l2_Ie'] = 0.1e-3 # The time constant of the l2 excitatory synapse
# parameters['tau_l2_Ii'] = 0.1e-3  # The time constant of the l2 inhibitory synapse
# parameters['v_l2_threshold'] = 10e-3  # The threshold of the l2 neuron
# parameters['refrac_l2'] = 0.1e-3  # The refractory period of the l2 neuron
# parameters['I_minimum_l2'] = 0.012e-12  # The minimum CCUR current fed to the l2 neuron
# # parameters['I_minimum_l2'] = 1 * pA
# parameters['I_step_l2'] = 0.0015e-12  # The variation of CCUR current fed to the l2 neuron
# parameters['reset_l2'] = 0
# parameters['trials_per_stimulus'] = trial_n
parameters['neurons_n'] = len(freqs)
parameters['frequencies'] = freqs
# parameters['TDE_to_l2_current'] = 1e-12
parameters['clock_sim'] = clock_sim
parameters['device'] = device

parameters['TDE_to_Osc_current'] = nn.Parameter(torch.Tensor([parameters['TDE_to_Osc_current']]), requires_grad=True)  # The current from the TDE to the OSC
parameters['TDE_to_HP0_current'] = nn.Parameter(torch.Tensor([parameters['TDE_to_HP0_current']]), requires_grad=True)  # The current from the TDE to the first stage of HP
# parameters['HP0_to_HP1_current'] = 1e-12  # The current from the first stage of HP to the second stage
# parameters['HP1_to_HP2_current'] = 1e-12  # The current from the second stage of HP to the third stage
# parameters['HP2_to_nWTA_current'] = 1e-12  # The current from the third stage of HP to the WTA
# parameters['HP2_to_nWTA_leftc'] = 0.05e-12  # The current from the third stage of HP to the WTA on the left
# parameters['HP2_to_nWTA_rightc'] = 0.05e-12  # The current from the third stage of HP to the WTA on the right
# parameters['nWTA_to_gWTA_current'] = 1e-12  # The current from the WTA to the Global Inh mechanism
# parameters['gWTA_to_nWTA_current'] = 1e-12 # The current from the Global Inh mechanism to the WTA
# parameters['Fake_to_nWTA_current'] = 0  # The current from the Input to the WTA
# parameters['Fake_to_gWTA_current'] = 0  # The current from the Input to the Global Inh mechanism
# parameters['in_to_TDE_current'] = 0
parameters['device'] = device
parameters['tot_time'] = sim_time



h1 = nn.Parameter(torch.Tensor([-1e-12,-1e-12,-1e-12]),requires_grad=False)

def generate_one_freq(f_ix,freq):
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
    return time_array_noisy,freq_array,trial_array,labels

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

dl_train = DataLoader(ds_train, batch_size=5, shuffle=True, num_workers=4, pin_memory=True)
dl_test = DataLoader(ds_test, batch_size=5, shuffle=True, num_workers=4, pin_memory=True)

softmin_fn = nn.Softmin(dim=1)
# loss_fn = nn.NLLLoss()
loss_fn = nn.CrossEntropyLoss()
torch.autograd.set_detect_anomaly(True)
currents = []
loss = []
acc = []


def objective(config):
    sPLL_array = sPLL(config).to(device)
    # L2_array = LIF_neuron_l2(config).to(device)
    optimizer = torch.optim.Adamax(sPLL_array.parameters(), lr=1e-16, betas=(0.9, 0.995))

    for i in range(100):
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

                # plt.plot([freqs[int(j)] for j in y_local.clone().detach().cpu().numpy()], mymin, '.')
                # plt.show()
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
# search_space = {"C_TDE": tune.loguniform(1e-4, 1e-2)}
# algo = tune.search._import_optuna_search()
#
# # 3. Start a Tune run that maximizes mean accuracy and stops after 5 iterations.
# tuner = tune.Tuner(
#     objective(config=parameters),
#     tune_config=tune.TuneConfig(
#         metric="mean_accuracy",
#         mode="max",
#         search_alg=algo,
#     ),
#     run_config=air.RunConfig(
#         stop={"training_iteration": 5},
#     ),
#     param_space=parameters,
# )
# results = tuner.fit()
    # # plt.title('TDE spikes')
    # # plt.eventplot(spk_tde_events[:,:,0])
    # # plt.figure()
    # plt.title('CCO spikes')
    # plt.eventplot(torch.where(spk_lif[:,:,0]))
    # plt.show()
    # print('ciao')
print('ciao')