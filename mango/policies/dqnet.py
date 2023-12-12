from dataclasses import InitVar, dataclass, field
from typing import Any
import numpy as np
import torch

from mango import spaces
from mango.protocols import ActType, ObsType, TensorTransitionLists, TrainInfo, Policy
from mango.neuralnetworks.networks import ConvEncoder


class DQNetPolicy(Policy):
    def __init__(
        self,
        action_space: spaces.Discrete,
        net_params: dict[str, Any] = dict(),
        lr: float = 1e-3,
        gamma: float = 0.9,
        tau: float = 0.005,
    ):
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
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
        self.net.eval()
        qvals = self.net(tensor_obs)
        if not batched:
            qvals = qvals.squeeze(0)
        return qvals

    def train(self, transitions: TensorTransitionLists) -> TrainInfo:
        if not transitions:
            raise ValueError("transitions list cannot be empty")
        self.net.train()
        self.target_net.train()
        td = self.temporal_difference(transitions)
        loss = torch.nn.functional.smooth_l1_loss(td, torch.zeros_like(td))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_models()
        return TrainInfo(loss=loss.item(), td=td.detach().cpu().numpy())

    def update_models(self):
        # soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # stochastic weight averaging of qnet
        # if self.ema_model is None:
        # self.ema_model = torch.optim.swa_utils.AveragedModel(self.net)
        # self.ema_model.update_parameters(self.net)

    def temporal_difference(self, transitions: TensorTransitionLists) -> torch.Tensor:
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
        qval_start: torch.Tensor = self.net(start_obs)
        qval_sampled_action = torch.gather(qval_start, 1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qval_next: torch.Tensor = self.net(next_obs)
            best_next_action = qval_next.argmax(dim=1, keepdim=True)
            best_qval_next = torch.gather(self.target_net(next_obs), 1, best_next_action).squeeze(1)

            # mango specific termination qvals
            best_qval_next[terminated_option] = 0.5 * 1.0 / self.gamma
            best_qval_next[truncated_option] = 0.5 * 1.0 / self.gamma
            best_qval_next[terminated] = 0.0

        return qval_sampled_action - rewards - self.gamma * best_qval_next
