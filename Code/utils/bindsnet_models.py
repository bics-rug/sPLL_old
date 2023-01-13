

from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Iterable, Optional, Union
from bindsnet.network.topology import Connection
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from torch.nn import Module, Parameter

import torch

from bindsnet.network.nodes import Nodes

class Connection_TDE(Connection):
    # language=rst
    """
    Specifies synapses between one or two populations of neurons.
    author: Nicoletta Risi n.risi@rug.nl
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        tau_fac: float = 10.0,
        tau_trig: float = 10.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """

        """
        The assumption here for this type of connection which implements the TDE is that the first
        input is always the facilitatory synapse and the second one is the trigger one.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)

        if isinstance(self.target, CSRMNodes):
            self.s_w = None

        self.dt = kwargs.get("dt", None)
        self.tau_fac = tau_fac
        self.tau_trig = tau_trig
        self.p = torch.zeros(1, target.shape[0])
        self.f = 0 # trace for the facilitatory synapse

        self.alpha = np.exp(-self.dt/self.tau_fac)
        self.beta = np.exp(-self.dt/self.tau_trig)

        # For debug:
        print('Shape source:', source.shape)
        print('Shape w:', self.w.shape)
        self.run_time = 0 # To count how many times the compute function is executed

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """
        # Compute multiplication of spike activations by weights and add bias.
        self.run_time += 1

        # Decay and update facilitatory synapse:
        self.f *= self.alpha
        self.f += s[0][0] #TODO: Check this update

        # Decay and update trigger synapse:
        post = self.p * self.beta
        post += s.view(s.size(0), -1).float() @ (self.w*self.f)
        self.p = post
        # The first conn is the one to the facilitatory input.
        # The weight of the second connection depends on the trace of the fac input

        return post.view(s.size(0), *self.target.shape)

    def compute_window(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """ """
        self.run_time -=1
        if self.s_w == None:
            # Construct a matrix of shape batch size * window size * dimension of layer
            self.s_w = torch.zeros(
                self.target.batch_size, self.target.res_window_size, *self.source.shape
            )

        # Add the spike vector into the first in first out matrix of windowed (res) spike trains
        self.s_w = torch.cat((self.s_w[:, 1:, :], s[:, None, :]), 1)

        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
            )
        else:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
                + self.b
            )

        return post.view(
            self.s_w.size(0), self.target.res_window_size, *self.target.shape
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()