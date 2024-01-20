from typing import Any, NamedTuple, Optional
from . import spaces, utils
from .protocols import AbstractAction, Policy
from .protocols import Environment, ObsType, ActType, Transition


class MangoEnv:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    @property
    def action_space(self) -> spaces.Discrete:
        return self.environment.action_space

    def step(self, comand: ActType, randomness=0.0) -> Transition:
        start_obs = self.obs
        self.obs, reward, term, trunc, info = self.environment.step(int(comand))
        return Transition(start_obs, comand, self.obs, float(reward), term, trunc)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        self.obs, info = self.environment.reset(seed=seed, options=options)
        return self.obs, info


class MangoLayer(MangoEnv):
    def __init__(
        self,
        environment: MangoEnv,
        abstract_actions: list[AbstractAction],
        policies: list[Policy],
    ):
        if len(abstract_actions) != len(policies):
            raise ValueError("Mismatch between number of abstract actions and policies")
        self.environment = environment
        self.abstract_actions = abstract_actions
        self.policies = policies

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.abstract_actions))

    def step(self, comand: ActType, randomness=0.0, max_loops=1) -> Transition:
        steps: list[Transition] = []
        while True:
            obs_masked = self.abstract_actions[comand].mask(self.environment.obs)
            action = self.policies[comand].get_action(obs_masked, randomness)
            steps.append(self.environment.step(action))

            if sum([(steps[-1].next_obs == step.start_obs).all() for step in steps]) > max_loops:
                steps[-1] = steps[-1]._replace(truncated=True)

            if steps[-1].terminated or steps[-1].truncated:
                break
            if self.abstract_actions[comand].beta(steps):
                break

        return Transition.from_steps(comand, steps)


class Agent:
    def __init__(
        self,
        environment: MangoEnv,
        policy: Policy,
    ) -> None:
        self.environment = environment
        self.policy = policy

    def run_episode(self, randomness: float = 0.0, max_loops=1) -> Transition:
        self.environment.reset()

        steps: list[Transition] = []
        while True:
            action = self.policy.get_action(self.environment.obs, randomness)
            steps.append(self.environment.step(action, randomness))

            # early stop in loops
            if sum([(steps[-1].next_obs == step.start_obs).all() for step in steps]) > max_loops:
                steps[-1] = steps[-1]._replace(truncated=True)

            if steps[-1].terminated or steps[-1].truncated:
                break
        return Transition.from_steps(None, steps)
