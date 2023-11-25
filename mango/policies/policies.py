from dataclasses import InitVar, dataclass, field
import random
from typing import Any, Optional, Protocol, Sequence
import copy
import numpy as np
import torch

from .. import spaces
from ..neuralnetworks.networks import ConvEncoder, squash
from ..utils import TensorTransitionLists, Transition, ObsType, ActType


class Policy(Protocol):
    action_space: spaces.Discrete

    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...

    def train(self, transitions: TensorTransitionLists) -> float | None:
        ...


@dataclass(eq=False, slots=True, repr=True)
class RandomPolicy(Policy):
    action_space: spaces.Discrete

    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        return ActType(int(self.action_space.sample()))

    def train(self, transitions: TensorTransitionLists) -> torch.Tensor | None:
        return None


@dataclass(eq=False, slots=True, repr=True)
class DQnetPolicy(Policy):
    action_space: spaces.Discrete

    net_params: InitVar[dict[str, Any]] = dict()
    lr: InitVar[float] = 1e-3
    gamma: float = field(default=0.9, repr=False)
    tau: float = field(default=0.05, repr=False)

    net: torch.nn.Module = field(init=False, repr=False)
    target_net: torch.nn.Module = field(init=False, repr=False)
    optimizer: torch.optim.Optimizer = field(init=False, repr=False)
    device: torch.device = field(init=False, repr=False)
    compile: bool = field(init=False, repr=False, default=True)

    def __post_init__(self, net_params, lr):
        self.net = ConvEncoder(None, int(self.action_space.n), **net_params)
        self.target_net = ConvEncoder(None, int(self.action_space.n), **net_params)
        self.optimizer = torch.optim.RAdam(self.net.parameters(recurse=True), lr=lr)
        self.device = next(self.net.parameters()).device

    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        with torch.no_grad():
            action_log_prob = self.qvalues(obs)
            if randomness == 0.0:
                return ActType(int(action_log_prob.argmax().item()))
            elif randomness == 1.0:
                return ActType(int(self.action_space.sample()))
            else:
                temperture = -np.log(1 - randomness) / 2
                probs = torch.softmax(action_log_prob / temperture, dim=-1)
                return ActType(int(torch.multinomial(probs, 1).item()))

    def qvalues(self, obs: ObsType) -> torch.Tensor:
        if self.net.training:
            self.net.eval()
        tensor_obs = torch.as_tensor(obs, dtype=torch.get_default_dtype(), device=self.device)
        qvals = self.net(tensor_obs.unsqueeze(0)).squeeze(0)
        return qvals

    def train(self, transitions: TensorTransitionLists) -> float | None:
        if not transitions:
            return None
        self.net.train()
        self.target_net.train()
        loss = self.compute_loss(transitions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_net()
        return loss.item()

    def update_target_net(self):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def compute_loss(self, transitions: TensorTransitionLists) -> torch.Tensor:
        # unpack sequence of transitions into sequence of its components
        start_obs, actions, next_obs, rewards, terminated, truncated, info = transitions
        start_obs = start_obs.to(dtype=torch.get_default_dtype(), device=self.device)
        actions = actions.to(dtype=torch.int64, device=self.device)
        next_obs = next_obs.to(dtype=torch.get_default_dtype(), device=self.device)
        rewards = rewards.to(dtype=torch.get_default_dtype(), device=self.device)
        terminated = terminated.to(dtype=torch.bool, device=self.device)
        truncated = truncated.to(dtype=torch.bool, device=self.device)
        terminated_option = torch.as_tensor(
            np.array([i["mango:terminated"] for i in info]), dtype=torch.bool, device=self.device
        )
        truncated_option = torch.as_tensor(
            np.array([i["mango:truncated"] for i in info]), dtype=torch.bool, device=self.device
        )

        # double DQN - use qnet to select best action, use target_net to evaluate it
        qval_start = self.net(start_obs)
        qval_sampled_action = torch.gather(qval_start, 1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qval_next = self.net(next_obs)
            best_next_action = qval_next.argmax(axis=1).unsqueeze(1)
            best_qval_next = torch.gather(self.target_net(next_obs), 1, best_next_action).squeeze(1)
            best_qval_next[terminated_option] = 1.0
            best_qval_next[truncated_option] = 1.0
            best_qval_next[terminated] = 0.0
        loss = torch.nn.functional.smooth_l1_loss(
            qval_sampled_action, rewards + self.gamma * best_qval_next
        )
        return loss
