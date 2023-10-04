import copy
from functools import partial
import numpy as np
import torch
from dataclasses import InitVar, dataclass, field
from typing import Any, Iterable, Optional, Protocol, TypeVar

from utils.neuralnetworks import NNFactory, OptimizableNN
from utils.spaces import SASRI, FlattenableSpace, SpaceType, TensorSpace, FiniteSpace


# region protocols

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Policy(Protocol[ObsType, ActType]):
    observation_space: SpaceType[ObsType]
    action_space: SpaceType[ActType]

    def __call__(self, state: ObsType) -> ActType:
        return self.get_action(state)

    def get_action(self, state: ObsType) -> ActType:
        ...

    def set_exploration_rate(self, exploration_rate: float):
        ...

    def train(self, sasri_list: Iterable[SASRI[ObsType, ActType]]):
        ...


class PolicyFactory(Protocol):
    def __call__(
        self,
        observation_space: SpaceType[ObsType],
        action_space: SpaceType[ActType],
    ) -> Policy[ObsType, ActType]:
        return self.make(observation_space, action_space)

    def make(
        self,
        observation_space: SpaceType[ObsType],
        action_space: SpaceType[ActType],
    ) -> Policy[ObsType, ActType]:
        ...


# endregion

# region random policy


@dataclass(frozen=True, eq=False)
class RandomPolicy(Policy[Any, ActType]):
    observation_space: SpaceType[Any]
    action_space: SpaceType[ActType]

    def get_action(self, state: Any) -> ActType:
        return self.action_space.sample()

    def train(self, sasri_list: Iterable[SASRI[Any, ActType]]):
        pass

    def set_exploration_rate(self, exploration_rate: float):
        pass


# endregion

# region qnet policy


@dataclass(frozen=True, eq=False)
class DiscreteQnetPolicy(Policy[np.ndarray, ActType]):
    observation_space: TensorSpace
    action_space: FiniteSpace[ActType]
    net_factory: InitVar[NNFactory[OptimizableNN]]
    loss_function = torch.nn.SmoothL1Loss()
    gamma: float = field(default=0.99, repr=False)
    exploration_rate: float = field(default=1.0, repr=False)
    train_cycles: int = field(default=1, repr=False)
    device: Optional[torch.device] = field(default=None, repr=False)

    net: OptimizableNN = field(init=False, repr=False)
    target_net: OptimizableNN = field(init=False, repr=False)
    refresh_timer: int = field(init=False, default=0, repr=False)
    loss_log: list[float] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self, net_factory: NNFactory[OptimizableNN]):
        net = net_factory(self.observation_space.shape, (len(self.action_space),))
        object.__setattr__(self, "net", net)
        object.__setattr__(self, "target_net", copy.deepcopy(net))

    def get_action(self, state: np.ndarray) -> ActType:
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_log_prob = self.net.forward(tensor_state, batched=False)
        action_prob = torch.softmax(action_log_prob / self.exploration_rate, dim=-1)
        action_idx = torch.multinomial(action_prob, num_samples=1)
        return self.action_space[int(action_idx.item())]

    def refresh_target(self, maxsteps: int = 10):
        if self.refresh_timer == maxsteps:
            object.__setattr__(self, "target_net", copy.deepcopy(self.net))
            object.__setattr__(self, "refresh_timer", 0)
        object.__setattr__(self, "refresh_timer", self.refresh_timer + 1)

    def set_exploration_rate(self, exploration_rate: float):
        object.__setattr__(self, "exploration_rate", exploration_rate + 1e-10)

    @property
    def loss_log_roll_mean(self) -> list[float]:
        return [
            float(np.mean(self.loss_log[i - 20 : i])) for i in range(20, len(self.loss_log))
        ]

    def compute_loss(
        self, sasri_list: Iterable[SASRI[np.ndarray, ActType]]
    ) -> torch.Tensor:
        start_states = [sasri.start_state for sasri in sasri_list]
        action_idxs = [self.action_space.index(sasri.action) for sasri in sasri_list]
        end_states = [sasri.end_state for sasri in sasri_list]
        rewards = [sasri.reward for sasri in sasri_list]

        tensor_start_states = torch.as_tensor(
            np.stack(start_states), dtype=torch.float32
        )
        tensor_action_idxs = torch.as_tensor(
            np.array(action_idxs), dtype=torch.int64
        ).unsqueeze(1)
        tensor_end_states = torch.as_tensor(np.stack(end_states), dtype=torch.float32)
        tensor_rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32)

        qvals = torch.gather(
            self.net(tensor_start_states), 1, tensor_action_idxs
        ).squeeze(1)
        qvals_target = self.target_net(tensor_end_states).detach().numpy().max(axis=1)

        loss = self.loss_function(qvals, tensor_rewards + self.gamma * qvals_target)
        return loss

    def train(self, sasri_list: Iterable[SASRI[np.ndarray, ActType]]):
        self.refresh_target()
        for _ in range(self.train_cycles):
            loss = self.compute_loss(sasri_list)
            self.loss_log.append(loss.item())
            self.net.optimization_step(loss)


@dataclass(frozen=True, eq=False)
class DiscreteQnetPolicyFactory(PolicyFactory):
    net_factory: NNFactory[OptimizableNN]

    def make(
        self, observation_space: TensorSpace, action_space: FiniteSpace[ActType]
    ) -> DiscreteQnetPolicy[ActType]:
        return DiscreteQnetPolicy(
            observation_space=observation_space,
            action_space=action_space,
            net_factory=self.net_factory,
        )


# endregion


@dataclass(frozen=True, eq=False)
class FlattenPolicyAdapter(Policy[Any, ActType]):
    observation_space: FlattenableSpace
    action_space: SpaceType[ActType]
    flat_policy: Policy[np.ndarray, ActType]

    def flatten(self, obs: Any) -> np.ndarray:
        return self.observation_space.flatten_obs(obs)

    def process_sasri(self, sasri: SASRI[Any, ActType]) -> SASRI[np.ndarray, ActType]:
        if sasri.preprocessed is None:
            sasri.preprocess(
                lambda sasri: SASRI.from_sasri_template(
                    template=sasri,
                    start_state=self.flatten(sasri.start_state),
                    end_state=self.flatten(sasri.end_state),
                )
            )
        return sasri.preprocessed  # type: ignore

    def get_action(self, state: Any) -> ActType:
        return self.flat_policy.get_action(self.flatten(state))

    def set_exploration_rate(self, exploration_rate: float):
        self.flat_policy.set_exploration_rate(exploration_rate)

    def train(self, sasri_list: Iterable[SASRI[np.ndarray, ActType]]):
        self.flat_policy.train([self.process_sasri(sasri) for sasri in sasri_list])


@dataclass(frozen=True, eq=False)
class FlattenSpacePolicyAdapterFactory(PolicyFactory):
    policy_factory: PolicyFactory

    def make(
        self, observation_space: FlattenableSpace, action_space: SpaceType[ActType]
    ) -> FlattenPolicyAdapter[ActType]:
        flat_obs_space = TensorSpace((observation_space.flat_dim,))
        return FlattenPolicyAdapter(
            observation_space=observation_space,
            action_space=action_space,
            flat_policy=self.policy_factory(flat_obs_space, action_space),
        )
