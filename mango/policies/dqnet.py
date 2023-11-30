from dataclasses import InitVar, dataclass, field
from typing import Any
import numpy as np
import torch

from mango import spaces
from mango.protocols import ActType, ObsType, TensorTransitionLists, Policy
from mango.neuralnetworks.networks import ConvEncoder


@dataclass(eq=False, slots=True, repr=True)
class DQNetPolicy(Policy):
    action_space: spaces.Discrete

    net_params: InitVar[dict[str, Any]] = dict()
    lr: InitVar[float] = 1e-3
    gamma: float = field(default=0.9, repr=False)
    tau: float = field(default=0.01, repr=False)

    net: torch.nn.Module = field(init=False, repr=False)
    target_net: torch.nn.Module = field(init=False, repr=False)
    ema_model: torch.nn.Module | None = field(init=False, repr=False, default=None)
    optimizer: torch.optim.Optimizer = field(init=False, repr=False)
    device: torch.device = field(init=False, repr=False)

    def __post_init__(self, net_params, lr):
        self.net = ConvEncoder(
            in_channels=None, out_features=int(self.action_space.n), **net_params
        ).train()
        self.target_net = ConvEncoder(
            in_channels=None, out_features=int(self.action_space.n), **net_params
        ).train()
        self.optimizer = torch.optim.RAdam(self.net.parameters(recurse=True), lr=lr)
        self.device = next(self.net.parameters()).device

    @classmethod
    def make(cls, action_space: spaces.Discrete, **kwargs) -> Policy:
        return cls(action_space, **kwargs)

    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        with torch.no_grad():
            action_log_prob = self.qvalues(obs)
            if randomness <= 0.0 + 1e-08:
                return int(action_log_prob.argmax().item())
            elif randomness >= 1.0 - 1e-08:
                return self.action_space.sample()
            else:
                temperture = -np.log2(1 - randomness) / 2
                probs = torch.softmax(action_log_prob / temperture, dim=-1)
                return int(torch.multinomial(probs, 1).item())

    def qvalues(self, obs: ObsType | torch.Tensor, batched=False) -> torch.Tensor:
        tensor_obs = torch.as_tensor(obs, dtype=torch.get_default_dtype(), device=self.device)
        if not batched:
            tensor_obs = tensor_obs.unsqueeze(0)
        if self.ema_model is None:
            qvals = self.net(tensor_obs)
            self.ema_model = torch.optim.swa_utils.AveragedModel(self.net).eval()
        else:
            qvals = self.ema_model(tensor_obs)
        if not batched:
            qvals = qvals.squeeze(0)
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
        self.update_models()
        return loss.item()

    def update_models(self):
        # soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # stochastic weight averaging of qnet
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.net)

    def compute_loss(self, transitions: TensorTransitionLists) -> torch.Tensor:
        # unpack sequence of transitions into sequence of its components
        start_obs = transitions.start_obs.to(dtype=torch.get_default_dtype(), device=self.device)
        actions = transitions.action.to(dtype=torch.int64, device=self.device)
        next_obs = transitions.next_obs.to(dtype=torch.get_default_dtype(), device=self.device)
        rewards = transitions.reward.to(dtype=torch.get_default_dtype(), device=self.device)
        terminated = transitions.terminated.to(dtype=torch.bool, device=self.device)
        truncated = transitions.truncated.to(dtype=torch.bool, device=self.device)

        terminated_option = np.array([i.get("mango:terminated", False) for i in transitions.info])
        truncated_option = np.array([i.get("mango:truncated", False) for i in transitions.info])
        terminated_option = torch.as_tensor(terminated_option, dtype=torch.bool, device=self.device)
        truncated_option = torch.as_tensor(truncated_option, dtype=torch.bool, device=self.device)

        # double DQN - use qnet to select best action, use target_net to evaluate it
        qval_start = self.net(start_obs)
        qval_sampled_action = torch.gather(qval_start, 1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qval_next = self.net(next_obs)
            best_next_action = qval_next.argmax(dim=1, keepdim=True)
            best_qval_next = torch.gather(self.target_net(next_obs), 1, best_next_action).squeeze(1)
            best_qval_next[terminated_option] = 1.0
            best_qval_next[truncated_option] = 1.0
            best_qval_next[terminated] = 0.0
        loss = torch.nn.functional.smooth_l1_loss(
            qval_sampled_action, rewards + self.gamma * best_qval_next
        )
        return loss
