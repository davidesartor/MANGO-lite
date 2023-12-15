from typing import Any, Sequence
import numpy as np
import torch

from mango import spaces
from mango.protocols import ActType, ObsType, TrainInfo, Policy, Transition
from mango.neuralnetworks.networks import ConvEncoder


def to_tensor(
    transitions: Sequence[Transition],
    obs_dtype=torch.get_default_dtype(),
    action_dtype=torch.int64,
    device=None,
):
    start_obs, action, next_obs, reward, terminated, truncated, info = zip(*transitions)
    start_obs = torch.as_tensor(np.stack(start_obs), dtype=obs_dtype, device=device)
    action = torch.as_tensor(np.array(action), dtype=action_dtype, device=device)
    next_obs = torch.as_tensor(np.stack(next_obs), dtype=obs_dtype, device=device)
    reward = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device)
    terminated = torch.as_tensor(np.array(terminated), dtype=torch.bool, device=device)
    return start_obs, action, next_obs, reward, terminated


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
            if randomness >= 1.0 - 1e-08:
                return int(torch.randint(0, int(self.action_space.n), (1,)).item())

            action_log_prob = self.qvalues(obs)
            if randomness <= 0.0 + 1e-08:
                return int(action_log_prob.argmax().item())

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

    def train(self, transitions: Sequence[Transition]) -> TrainInfo:
        if len(transitions) == 0:
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

    def temporal_difference(self, transitions: Sequence[Transition]) -> torch.Tensor:
        # unpack sequence of transitions into sequence of its components
        start_obs, actions, next_obs, rewards, term = to_tensor(transitions, device=self.device)

        # double DQN - use qnet to select best action, use target_net to evaluate it
        qval_start: torch.Tensor = self.net(start_obs)
        qval_sampled_action = torch.gather(qval_start, 1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qval_next: torch.Tensor = self.net(next_obs)
            best_next_action = qval_next.argmax(dim=1, keepdim=True)
            best_qval_next = torch.gather(self.target_net(next_obs), 1, best_next_action).squeeze(1)
            best_qval_next[term] = 0.0
        return qval_sampled_action - rewards - self.gamma * best_qval_next
