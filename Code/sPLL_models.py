import torch
import torch.nn as nn
from collections import namedtuple


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    # controls steepness of surrogate gradient
    scale = 20.0

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
        self.dt = parameters['clock_sim']
        self.device = parameters['device']
        # Osc specific
        self.tau_syn = parameters['tau_Osc_Ie']
        self.tau_neu = parameters['tau_Osc']
        self.gain_syn = parameters['TDE_to_Osc_current']
        self.Vr = parameters['reset_osc']
        self.I_minimum_osc = parameters['I_minimum_osc']
        self.I_step_osc = parameters['I_step_osc']
        self.refrac_Osc = parameters['refrac_Osc']
        self.steady_current = 0
        self.refr = self.refrac_Osc/self.dt

    def initialize_state(self,parameters):
        self.state = None
        self.bs = parameters['trials_per_stimulus']

    def forward(self, input):
        device = input.device
        if self.state is None:
            self.state = self.NeuronState(V = torch.ones((self.bs,self.n), device=device),
                                          I = torch.zeros((self.bs,self.n), device=device),
                                          count_refr = torch.zeros((self.bs,self.n), device=device))

        v = self.state.V
        i = self.state.I
        count_refr = self.state.count_refr

        i += -self.dt * i / self.tau_syn + self.gain_syn.item() * input

        v += self.dt * (-v / self.tau_neu + (i + self.steady_current) / self.C)

        spk = SurrGradSpike.apply(v - self.thr)

        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)

        v = (1 - spk) * v * (count_refr <= 0) + spk * self.Vr
        # print(V.shape)
        
        self.state = self.NeuronState(V=v, I=i,count_refr = count_refr)
        # del v, i, count_refr

        return spk


class TDE(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i_trg', 'i_fac','count_refr'])

    def __init__(self, parameters, device):
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
        device = input_trg.device
        if self.state is None:
            self.state = self.NeuronState(V = torch.zeros(input_trg.shape, device=device),
                                          i_trg = torch.zeros(input_trg.shape, device=device),
                                          i_fac = torch.zeros(input_trg.shape, device=device),count_refr = torch.zeros((self.bs,self.n), device=device))

        assert input_trg.shape == input_fac.shape, f"trg {input_trg.shape} fac {input_fac.shape}"

        v = self.state.V
        i_trg = self.state.i_trg
        count_refr = self.state.count_refr

        i_fac = input_fac * self.gain_fac + self.state.i_fac * (1-input_fac)
        i_fac += -self.dt / self.tau_fac * i_fac

        i_trg += -self.dt / self.tau_trg * i_trg
        i_trg += input_trg * self.gain_trg * i_fac * (i_fac > self.ifac_thr)
        # print('i_trg',i_trg,'i_fac',i_fac)

        v += self.dt * (- v / self.beta + i_trg / self.C)
        # print(v)

        spk = SurrGradSpike.apply(v - self.thr)
        # print('count_refr',count_refr)
        # print('spk',spk)

        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)
        v = (1 - spk) * v * (count_refr <= 0) + spk * self.Vr

        self.state = self.NeuronState(V=v, i_trg=i_trg, i_fac=i_fac, count_refr=count_refr)

        return spk


class sPLL(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i_trg', 'i_fac'])

    def __init__(self, parameters, device):
        super(sPLL, self).__init__()
        self.TDE = TDE(parameters, device).to(device)
        self.LIF = LIF_neuron(parameters).to(device)
        self.st_n = len(parameters['frequencies'])

        self.n_out = parameters['neurons_n']
        self.n_in = 1
        self.tot_time = int(parameters['tot_time']/parameters['clock_sim'])
        self.device = parameters['device']

        self.spike_record_LIF = []
        self.spike_record_TDE = []

        cur_list = torch.linspace(
            parameters['I_minimum_osc'], 
            parameters['I_minimum_osc'] + parameters['I_step_osc']*self.n_out, 
            self.n_out)
        self.tmp_current_list = nn.Parameter(cur_list, requires_grad=True)

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
        self.spikes_TDE_prev = self.spikes_TDE.clone()
