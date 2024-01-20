import copy
import numpy as np
import torch

from mango import spaces
from mango.protocols import ActType, ObsType, Policy, StackedTransitions, Trainer, TrainInfo


class DQNetPolicy(Policy):
    def __init__(self, net: torch.nn.Module, action_space: spaces.Discrete):
        self.action_space = action_space
        self.net = net

    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        with torch.no_grad():
            if randomness >= 1.0 - 1e-08:
                return int(torch.randint(0, int(self.action_space.n), ()))

            qvals: torch.Tensor = self.net(obs.unsqueeze(0)).squeeze(0)
            if randomness <= 0.0 + 1e-08:
                return int(qvals.argmax())

            temperture = -np.log2(1 - randomness) / 2
            probs = torch.softmax(qvals / temperture, dim=-1)
            return int(torch.multinomial(probs, 1).item())


class DQNetTrainer(Trainer):
    def __init__(self, net: torch.nn.Module, lr=1e-3, gamma=0.95, tau=0.001):
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.net = net

    def lazy_init(self):
        # need separate call when working with lazy modules
        self.target_net = copy.deepcopy(self.net).train()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self, transitions: StackedTransitions) -> TrainInfo:
        if len(transitions) == 0:
            raise ValueError(f"{self.__class__.__name__}.train received empty transitions list")
        if not hasattr(self, "optimizer"):
            self.lazy_init()

        self.net.train()
        td = self.temporal_difference(transitions)
        loss = torch.nn.functional.smooth_l1_loss(td, torch.zeros_like(td))
        self.optimization_step(loss)

        loss = loss.detach().to("cpu", non_blocking=True)
        td = td.detach().to("cpu", non_blocking=True)
        self.net.eval()
        return TrainInfo(loss=loss, td=td)

    def optimization_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # soft update of target network
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def temporal_difference(self, transitions: StackedTransitions) -> torch.Tensor:
        start_obs, action, next_obs, reward, terminated, truncated = transitions

        # double DQN - use qnet to select best action, use target_net to evaluate it
        qval_start = self.net(start_obs)
        qval_sampled_action = torch.gather(qval_start, 1, action.unsqueeze(1))
        with torch.no_grad():
            best_next_action = self.net(next_obs).argmax(dim=1, keepdim=True)
            qval_next = self.target_net(next_obs)
            best_qval_next = torch.gather(qval_next, 1, best_next_action)
            best_qval_next[terminated] = 0.0
        td = qval_sampled_action - reward.unsqueeze(1) - self.gamma * best_qval_next
        return td.squeeze(1)
