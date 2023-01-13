import torch
from torch import nn
from collections import namedtuple

from utils.utils_misc import update_progress
import numpy as np
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

    def initialize_state(self):
        self.state = None

    def forward(self, input):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones((self.bs,self.st_n,self.n), device=input.device),
                                          I=torch.zeros((self.bs,self.st_n,self.n), device=input.device),
                                          count_refr = torch.zeros((self.bs,self.st_n,self.n), device=input.device)
                                          )

        V = self.state.V
        I = self.state.I
        count_refr = self.state.count_refr

        I += -self.dt * I / self.tau_syn + self.gain_syn * input

        V += self.dt * (-V / self.tau_neu + (I + self.steady_current) / self.C)

        spk = activation(V - self.thr)
        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)
        V = (1 - spk) * V * (count_refr <= 0) + spk * self.Vr
        # print(V.shape)
        # print(V)
        self.state = self.NeuronState(V=V, I=I,count_refr = count_refr)
        del V,I,count_refr
        return spk

class LIF_only(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i_trg', 'i_fac'])

    def __init__(self, parameters):
        super(LIF_only, self).__init__()
        self.LIF = LIF_neuron(parameters)
        self.st_n = len(parameters['frequencies'])
        self.bs = parameters['trials_per_stimulus']
        self.tot_time = int(parameters['tot_time']/parameters['clock_sim'])
        self.n = parameters['neurons_n']
        self.device = parameters['device']
        # self.spikes_CCO = torch.zeros((self.bs,self.st_n,self.n), device=self.device).to_sparse()
        self.spike_record_LIF = []
        self.LIF.C = parameters['C_HP']
        self.LIF.thr = parameters['v_HP_threshold']
        self.LIF.tau_syn = parameters['tau_HP_Ie']
        self.LIF.tau_neu = parameters['tau_HP']
        self.LIF.gain_syn = torch.Tensor(parameters['TDE_to_HP0_current']*parameters['gain_weight_LIF'] + np.random.normal(-parameters['TDE_to_HP0_current']*parameters['std_weight_LIF']*1,parameters['TDE_to_HP0_current']*parameters['std_weight_LIF']*1, self.n)).to(self.device)
        # self.LIF.gain_syn = parameters['TDE_to_HP0_current']
        current_list = []
        for k in range(self.st_n):
            current_list.append([parameters['LIF_stable_current_value'] + parameters['LIF_stable_current_increase']*j for j in range(self.n)])
        self.LIF.steady_current = torch.tensor([current_list for i in range(self.bs)], device = self.device).to_sparse()
        self.LIF.Vr = 0
        self.LIF.I_minimum_osc = 0
        self.LIF.I_step_osc = 0
        self.LIF.refrac_Osc = parameters['refrac_HP']

    def initialize_state(self):
        self.LIF.initialize_state()
    def run(self,input):

        with torch.no_grad():
            input = input.to_dense()
            input = input.expand(-1, -1, -1, self.n)
            for t in range(0, input.shape[2]):

                update_progress((t + 1) / input.shape[2], 'simulating LIF')
                self.spikes_LIF = self.LIF(input[:,:,t,:])
                self.spike_record_LIF.append(self.spikes_LIF.to_sparse())
        self.spike_record_LIF= torch.stack(self.spike_record_LIF)


class TDE(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i_trg', 'i_fac','count_refr'])

    def __init__(self, parameters):
        super(TDE, self).__init__()
        self.C = parameters['C_TDE']
        self.thr = parameters['v_TDE_threshold']
        self.state = None
        self.bs = parameters['trials_per_stimulus']
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

    def initialize_state(self):
        self.state = None

    def forward(self, input_trg, input_fac):
        if self.state is None:
            self.state = self.NeuronState(V=torch.zeros((self.bs,self.st_n,self.n), device=input_trg.device),
                                          i_trg=torch.zeros((self.bs,self.st_n,self.n), device=input_trg.device),
                                          i_fac=torch.zeros((self.bs,self.st_n,self.n), device=input_trg.device),count_refr = torch.zeros((self.bs,self.st_n,self.n), device=input_trg.device)
                                          )
        assert input_trg.shape == input_fac.shape, "TRG and FAC should have same dimensions, while: shape trg: " + str(input_trg.shape) + ", shape fac: " + str(input_fac.shape)
        v = self.state.V
        i_trg = self.state.i_trg
        count_refr = self.state.count_refr
        i_fac = input_fac * self.gain_fac + self.state.i_fac * (1-input_fac)
        i_fac += -self.dt / self.tau_fac * i_fac
        # i_fac = i_fac * ((i_fac < 10) | (self.capped == False)) + 10 * ((i_fac >= 10) & (self.capped == True))

        i_trg += -self.dt / self.tau_trg * i_trg + input_trg * self.gain_trg * i_fac * (i_fac > self.ifac_thr)

        v += self.dt * (- v / self.beta + i_trg / self.C)
        # print(v)
        spk = activation(v - self.thr)
        # print('count_refr',count_refr)
        # print('spk',spk)
        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)
        v = (1 - spk) * v * (count_refr <= 0) + spk * self.Vr

        self.state = self.NeuronState(V=v, i_trg=i_trg, i_fac=i_fac,count_refr= count_refr)
        return spk


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

class HF(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i_trg', 'i_fac'])

    def __init__(self, parameters):
        super(HF, self).__init__()
        self.LIF = LIF_neuron(parameters)
        self.st_n = len(parameters['frequencies'])
        self.bs = parameters['trials_per_stimulus']
        self.tot_time = int(parameters['tot_time']/parameters['clock_sim'])
        self.n = parameters['neurons_n']
        self.device = parameters['device']
        # self.spikes_CCO = torch.zeros((self.bs,self.st_n,self.n), device=self.device).to_sparse()
        self.spike_record_LIF = []
        self.LIF.C = parameters['C_HP']
        self.LIF.thr = parameters['v_HP_threshold']
        self.LIF.tau_syn = parameters['tau_HP_Ie']
        self.LIF.tau_neu = parameters['tau_HP']
        self.LIF.gain_syn = parameters['TDE_to_HP0_current']
        self.LIF.Vr = 0
        self.LIF.I_minimum_osc = 0
        self.LIF.I_step_osc = 0
        self.LIF.refrac_Osc = parameters['refrac_HP']

    def initialize_state(self):
        self.LIF.initialize_state()
    def run(self,input):

        with torch.no_grad():
            input = input.to_dense()
            for t in range(0, input.shape[0]):

                update_progress((t + 1) / input.shape[0], 'simulating HF')
                self.spikes_LIF = self.LIF(input[t,:,:,:])
                self.spike_record_LIF.append(self.spikes_LIF.to_sparse())
        self.spike_record_LIF= torch.stack(self.spike_record_LIF)

class sPLL(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i_trg', 'i_fac'])

    def __init__(self, parameters):
        super(sPLL, self).__init__()
        self.TDE = TDE(parameters)
        self.LIF = LIF_neuron(parameters)
        self.st_n = len(parameters['frequencies'])
        self.bs = parameters['trials_per_stimulus']
        self.n_out = parameters['neurons_n']
        self.n_in = 1
        self.tot_time = int(parameters['tot_time']/parameters['clock_sim'])
        self.device = parameters['device']
        self.input_fac = torch.zeros((self.bs,self.st_n,self.n_out), device=self.device).to_sparse()
        self.spikes_TDE_prev = torch.zeros((self.bs,self.st_n,self.n_out), device=self.device).to_sparse()
        self.spikes_TDE = torch.zeros((self.bs,self.st_n,self.n_out), device=self.device).to_sparse()
        self.spikes_LIF = torch.zeros((self.bs,self.st_n,self.n_out), device=self.device).to_sparse()
        self.i_trg = torch.zeros((self.bs,self.st_n,self.n_out), device=self.device).to_sparse()
        self.i_fac = torch.zeros((self.bs,self.st_n,self.n_out), device=self.device).to_sparse()
        self.i_syn = torch.zeros((self.bs,self.st_n,self.n_out), device=self.device).to_sparse()
        self.spike_record_LIF = []
        self.spike_record_TDE = []
        self.linear = nn.Linear(self.n_in, self.n_out, bias=False).to(self.device)
        current_list = []
        for k in range(self.st_n):
            current_list.append([parameters['I_minimum_osc'] + parameters['I_step_osc']*j for j in range(self.n_out)])
        self.LIF.steady_current = torch.tensor([current_list for i in range(self.bs)], device = self.device).to_sparse()
    def initialize_state(self):
        self.LIF.initialize_state()
        self.TDE.initialize_state()

    def run(self, input):
        import gc
        # print('simulating...')
        with torch.no_grad():
            input = input.expand(-1,-1,-1,self.n_out)
            for t in range(0, input.shape[2]):
                update_progress((t+1)/input.shape[2],'simulating sPLL')
                #
                # print('time:', t)
                # torch.cuda.empty_cache()

                # gc.collect()
                self.spikes_LIF = self.LIF(self.spikes_TDE_prev)

                self.spikes_TDE = self.TDE(input[:,:,t,:], self.spikes_LIF)

                self.i_trg = self.TDE.state.i_trg
                self.i_fac = self.TDE.state.i_fac
                self.i_syn = self.LIF.state.I
                self.spikes_TDE_prev = self.spikes_TDE.clone()

                self.spike_record_LIF.append(self.spikes_LIF.to_sparse())
                self.spike_record_TDE.append(self.spikes_TDE.to_sparse())
                # print(self.spike_record_LIF)
        self.spike_record_LIF= torch.stack(self.spike_record_LIF)
        self.spike_record_TDE= torch.stack(self.spike_record_TDE)
