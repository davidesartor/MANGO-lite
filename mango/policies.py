from dataclasses import InitVar, dataclass, field
from typing import Any, Protocol, Sequence, SupportsFloat, TypeVar
import copy
import numpy as np
import numpy.typing as npt
import torch

from gymnasium import spaces
from .neuralnetworks.networks import ConvEncoder
from .utils import Transition


class Policy(Protocol):
    def get_action(self, state: npt.NDArray) -> int:
        ...

    def train(self, transitions: Sequence[Transition]):
        ...


@dataclass(eq=False, slots=True)
class RandomPolicy(Policy):
    action_space: spaces.Discrete

    def get_action(self, state: Any) -> int:
        return int(self.action_space.sample())

    def train(self, transitions: Sequence[Transition]):
        pass


@dataclass(eq=False, slots=True)
class DQnetPolicy(Policy):
    action_space: spaces.Discrete

    loss_function = torch.nn.SmoothL1Loss()
    gamma: float = field(default=0.99, repr=False)
    train_cycles: int = field(default=1, repr=False)
    refresh_timer: tuple[int, int] = field(default=(0, 1), repr=False)

    exploration_rate: float = field(init=False, default=1.0, repr=False)
    loss_log: list[float] = field(init=False, default_factory=list, repr=False)
    net: ConvEncoder = field(init=False, repr=False)
    target_net: ConvEncoder = field(init=False, repr=False)
    optimizer: torch.optim.Optimizer = field(init=False, repr=False)

    def __post_init__(self):
        self.net = ConvEncoder(in_channels=None, out_features=int(self.action_space.n))
        self.target_net = ConvEncoder(
            in_channels=None, out_features=int(self.action_space.n)
        )
        self.optimizer = torch.optim.Adam(params=self.net.parameters(recurse=True))

    def get_action(self, state: npt.NDArray) -> int:
        self.net.eval()
        tensor_state = torch.as_tensor(state, dtype=torch.float32)
        action_log_prob = self.net.forward(tensor_state.unsqueeze(0))
        action_prob = torch.softmax(action_log_prob / self.exploration_rate, dim=-1)
        action = torch.multinomial(action_prob, num_samples=1)
        return int(action.item())

    def train(self, transitions: Sequence[Transition]):
        self.net.train()
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

    def compute_loss(self, transitions: Sequence[Transition]) -> torch.Tensor:
        # unpack sequence of transitions into sequence of its components
        start_states, action_idxs, next_states, rewards, *_ = zip(*transitions)

        start_states = torch.as_tensor(np.stack(start_states), dtype=torch.float32)  # type: ignore
        actions = torch.as_tensor(np.array(action_idxs), dtype=torch.int64).unsqueeze(1)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)  # type: ignore
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32)

        qvals = torch.gather(self.net(start_states), 1, actions).squeeze(1)
        qvals_target = self.target_net(next_states).detach().numpy().max(axis=1)
        loss = self.loss_function(qvals, rewards + self.gamma * qvals_target)
        return loss
