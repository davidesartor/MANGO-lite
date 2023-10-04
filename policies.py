from dataclasses import InitVar, dataclass, field
from typing import Any, Sequence, TypeVar
import copy
import numpy as np
import torch

from . import spaces
from .neuralnetworks.networks import LazyConvEncoder
from .protocols import Policy, Transition

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass(frozen=True, eq=False)
class RandomPolicy(Policy[Any, ActType]):
    observation_space: InitVar[spaces.Space[Any]]
    action_space: spaces.Space[ActType]

    def set_exploration_rate(self, exploration_rate: float):
        pass

    def get_action(self, state: Any) -> ActType:
        return self.action_space.sample()

    def train(self, transitions: Sequence[Transition]):
        pass


Array3d = np.ndarray[tuple[int, int, int], Any]


@dataclass(eq=False, repr=False)
class DQnetPolicy(Policy[Array3d, int]):
    observation_space: spaces.Box
    action_space: spaces.Discrete

    loss_function = torch.nn.SmoothL1Loss()
    exploration_rate = 1.0
    gamma: float = 0.99
    train_cycles: int = 1
    refresh_timer: tuple[int, int] = (0, 10)

    loss_log: list[float] = field(init=False, default_factory=list)
    net: LazyConvEncoder = field(init=False)
    target_net: LazyConvEncoder = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self):
        squeezed_dim = torch.zeros(self.observation_space.shape).squeeze().ndim
        if self.observation_space.shape[0] == 1:
            squeezed_dim += 1

        kernel_size = (3,) * (len(self.observation_space.shape) - 1)
        self.net = LazyConvEncoder(
            in_shape=self.observation_space.shape, out_features=int(self.action_space.n)
        )
        self.optimizer = torch.optim.Adam(params=self.net.parameters(recurse=True))
        self.target_net = copy.deepcopy(self.net)

    def set_exploration_rate(self, exploration_rate: float):
        self.exploration_rate = exploration_rate + 1e-8

    def get_action(self, state: Array3d) -> int:
        self.net.eval()
        tensor_state = torch.as_tensor(state, dtype=torch.float32)
        action_log_prob = self.net.forward(tensor_state.unsqueeze(0))
        action_prob = torch.softmax(action_log_prob / self.exploration_rate, dim=-1)
        action = torch.multinomial(action_prob, num_samples=1)
        return int(action.item())

    def train(self, transitions: Sequence[Transition[np.ndarray, int]]):
        for _ in range(self.train_cycles):
            loss = self.compute_loss(transitions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_log.append(loss.item())
        self.update_target()

    def update_target(self):
        t, tmax = self.refresh_timer
        t = (t + 1) % tmax
        self.refresh_timer = (t, tmax)

        if t == 0:
            self.target_net = copy.deepcopy(self.net)

    def compute_loss(
        self, transitions: Sequence[Transition[np.ndarray, int]]
    ) -> torch.Tensor:
        start_states = torch.as_tensor(
            np.stack([t.start_state for t in transitions]), dtype=torch.float32
        )
        action_idxs = torch.as_tensor(
            np.array([t.action for t in transitions]), dtype=torch.int64
        ).unsqueeze(1)
        end_states = torch.as_tensor(
            np.stack([t.next_state for t in transitions]), dtype=torch.float32
        )
        rewards = torch.as_tensor(
            np.array([t.reward for t in transitions]), dtype=torch.float32
        )
        qvals = torch.gather(self.net(start_states), 1, action_idxs).squeeze(1)
        qvals_target = self.target_net(end_states).detach().numpy().max(axis=1)
        loss = self.loss_function(qvals, rewards + self.gamma * qvals_target)
        return loss
