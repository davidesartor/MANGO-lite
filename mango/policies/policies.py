from dataclasses import InitVar, dataclass, field
from typing import Any, Optional, Protocol, Sequence
import copy
import numpy as np
import torch

from .. import spaces
from .neuralnetworks.networks import ConvEncoder
from ..utils import Transition, ObsType, ActType


class Policy(Protocol):
    action_space: spaces.Discrete

    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...

    def train(self, transitions: Sequence[Transition]) -> float | None:
        ...


@dataclass(eq=False, slots=True, repr=True)
class RandomPolicy(Policy):
    action_space: spaces.Discrete

    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        return ActType(int(self.action_space.sample()))

    def train(self, transitions: Sequence[Transition]) -> float | None:
        return None


@dataclass(eq=False, slots=True, repr=True)
class DQnetPolicy(Policy):
    action_space: spaces.Discrete

    net_params: InitVar[dict[str, Any]] = dict()
    lr: InitVar[float] = 1e-4
    gamma: float = field(default=0.5, repr=False)
    refresh_timer: tuple[int, int] = field(default=(0, 10), repr=False)

    net: ConvEncoder = field(init=False, repr=False)
    target_net: ConvEncoder = field(init=False, repr=False)
    optimizer: torch.optim.Adam = field(init=False, repr=False)
    device: torch.device = field(init=False, repr=False, default=torch.device("cpu"))

    def __post_init__(self, net_params, lr):
        self.net = ConvEncoder(None, int(self.action_space.n), **net_params)
        self.target_net = ConvEncoder(None, int(self.action_space.n), **net_params)
        self.optimizer = torch.optim.Adam(self.net.parameters(recurse=True), lr=lr)
        self.device = next(self.net.parameters()).device

    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        self.net.eval()
        with torch.no_grad():
            tensor_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action_log_prob = self.net.forward(tensor_obs.unsqueeze(0))
            if randomness > 0.0:
                probs = torch.softmax(action_log_prob / randomness, dim=1)
                return ActType(int(torch.multinomial(probs, num_samples=1).item()))
            else:
                return ActType(int(action_log_prob.argmax().item()))

    def train(self, transitions: Sequence[Transition]) -> float | None:
        if not transitions:
            return None
        self.net.train()
        loss = self.compute_loss(transitions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_net()
        return float(loss.item())

    def update_target_net(self):
        t, tmax = self.refresh_timer
        t = (t + 1) % tmax
        self.refresh_timer = (t, tmax)
        if t == 0:
            self.target_net = copy.deepcopy(self.net)

    def compute_loss(self, transitions: Sequence[Transition]) -> torch.Tensor:
        # unpack sequence of transitions into sequence of its components
        start_obs, actions, next_obs, rewards, terminated, *_ = zip(*transitions)

        start_obs = torch.as_tensor(np.stack(start_obs), dtype=torch.float32, device=self.device)  # type: ignore
        actions = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(1)
        next_obs = torch.as_tensor(np.stack(next_obs), dtype=torch.float32, device=self.device)  # type: ignore
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(np.array(terminated), dtype=torch.bool, device=self.device)

        qvals = torch.gather(self.net(start_obs), 1, actions).squeeze(1)
        with torch.no_grad():
            qvals_target = self.target_net(next_obs).max(axis=1)[0] * (~terminated)
        return torch.nn.functional.smooth_l1_loss(
            qvals, rewards + self.gamma * qvals_target
        )
