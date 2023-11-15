from dataclasses import InitVar, dataclass, field
import re
from typing import Any, Protocol, Sequence
import copy
import numpy as np
import numpy.typing as npt
import torch
import gymnasium as gym

from .neuralnetworks.networks import ConvEncoder
from .utils import Transition


class Policy(Protocol):
    action_space: gym.spaces.Discrete

    def get_action(self, state: npt.NDArray, randomness: float = 0.0) -> int:
        ...

    def train(self, transitions: Sequence[Transition]) -> float:
        ...


@dataclass(eq=False, slots=True)
class RandomPolicy(Policy):
    action_space: gym.spaces.Discrete

    def get_action(self, state: Any, randomness: float = 0.0) -> int:
        return int(self.action_space.sample())

    def train(self, transitions: Sequence[Transition]):
        pass


@dataclass(eq=False, slots=True, repr=False)
class DQnetPolicy(Policy):
    action_space: gym.spaces.Discrete

    loss_function = torch.nn.SmoothL1Loss()
    gamma: float = 0.99
    lr: InitVar[float] = 1e-4
    refresh_timer: tuple[int, int] = (0, 10)

    net: ConvEncoder = field(init=False)
    target_net: ConvEncoder = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, lr: float):
        self.net = ConvEncoder(in_channels=None, out_features=int(self.action_space.n))
        self.target_net = ConvEncoder(
            in_channels=None, out_features=int(self.action_space.n)
        )
        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(recurse=True), lr=lr
        )

    def get_action(self, state: npt.NDArray, randomness: float = 0.0) -> int:
        self.net.eval()
        with torch.no_grad():
            tensor_state = torch.as_tensor(state, dtype=torch.float32)
            action_log_prob = self.net.forward(tensor_state.unsqueeze(0))
            if randomness > 0.0:
                probs = torch.softmax(action_log_prob / randomness, dim=1)
                return int(torch.multinomial(probs, num_samples=1).item())
            else:
                return int(action_log_prob.argmax().item())

    def train(self, transitions: Sequence[Transition]) -> float | None:
        if not transitions:
            return None
        self.net.train()
        loss = self.compute_loss(transitions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target()
        return float(loss.item())

    def update_target(self):
        t, tmax = self.refresh_timer
        t = (t + 1) % tmax
        self.refresh_timer = (t, tmax)
        if t == 0:
            self.target_net = copy.deepcopy(self.net)

    def compute_loss(self, transitions: Sequence[Transition]) -> torch.Tensor:
        # unpack sequence of transitions into sequence of its components
        start_states, actions, next_states, rewards, terminated, *_ = zip(*transitions)

        start_states = torch.as_tensor(np.stack(start_states), dtype=torch.float32)  # type: ignore
        actions = torch.as_tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)  # type: ignore
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32)
        terminated = torch.as_tensor(np.array(terminated), dtype=torch.bool)

        qvals = torch.gather(self.net(start_states), 1, actions).squeeze(1)
        with torch.no_grad():
            qvals_target = self.target_net(next_states).max(axis=1)[0] * (~terminated)
        loss = self.loss_function(qvals, rewards + self.gamma * qvals_target)
        return loss

    def __repr__(self):
        return f"{self.__class__.__name__}(action_space={self.action_space})"
