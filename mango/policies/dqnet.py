import copy
from typing import Any, Sequence
import numpy as np
import torch

from mango import spaces
from mango.protocols import ActType, ObsType, Policy, StackedTransitions, Trainer, Transition
from mango.neuralnetworks.networks import ConvEncoder


class DQNetPolicy(Policy):
    def __init__(self, net: torch.nn.Module, action_space: spaces.Discrete):
        self.action_space = action_space
        self.net = net

    @classmethod
    def make(
        cls, action_space: spaces.Discrete, lr=1e-3, gamma=0.95, tau=0.005, **net_kwargs
    ) -> tuple[Policy, Trainer]:
        net = ConvEncoder(in_channels=None, out_features=int(action_space.n), **net_kwargs).eval()
        trainer = DQNetTrainer(net, lr=lr, gamma=gamma, tau=tau)
        return cls(net, action_space), trainer

    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        with torch.no_grad():
            if randomness >= 1.0 - 1e-08:
                return torch.randint(0, int(self.action_space.n), ())

            qvals: torch.Tensor = self.net(obs.unsqueeze(0)).squeeze(0)
            if randomness <= 0.0 + 1e-08:
                return qvals.argmax()

            temperture = -np.log2(1 - randomness) / 2
            probs = torch.softmax(qvals / temperture, dim=-1)
            return torch.multinomial(probs, 1)


class DQNetTrainer(Trainer):
    def __init__(self, net: torch.nn.Module, lr=1e-3, gamma=0.95, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.net = net
        self.target_net = copy.deepcopy(net).train()
        self.optimizer = torch.optim.RAdam(self.net.parameters(recurse=True), lr=lr)

    def train(self, transitions: StackedTransitions) -> float:
        if len(transitions) == 0:
            raise ValueError(f"{self.__class__.__name__}.train received empty transitions list")

        self.net.train()
        td = self.temporal_difference(transitions)
        loss = torch.nn.functional.smooth_l1_loss(td, torch.zeros_like(td))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        self.net.eval()
        return loss.item()

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def temporal_difference(self, transitions: StackedTransitions) -> torch.Tensor:
        # double DQN - use qnet to select best action, use target_net to evaluate it
        qval_start = self.net(transitions.start_obs)
        qval_sampled_action = torch.gather(qval_start, 1, transitions.action.unsqueeze(1))
        with torch.no_grad():
            best_next_action = self.net(transitions.next_obs).qval_next.argmax(dim=1, keepdim=True)
            qval_next = self.target_net(transitions.next_obs)
            best_qval_next = torch.gather(qval_next, 1, best_next_action)
            best_qval_next[transitions.terminated] = 0.0
        return qval_sampled_action - transitions.reward.unsqueeze(1) - self.gamma * best_qval_next
