from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map

def generate_map(size=8, p=0.8, mirror=False, random_start=False):
    env_description = generate_random_map(size=size // 2 if mirror else size, p=p)
    if random_start:
        env_description = list(map(lambda row: row.replace("F", "S"), env_description)) 
    if mirror:
        env_description = [row[::-1]+ row for row in env_description[::-1] + env_description]
    return env_description

def make_custom_frozen_lake_env():
    

class CustomFrozenLakeEnv(gym.Wrapper):
    metadata = {
        "state_modes": ["agent_pos", "one_hot", "rgb_array", "flattened_one_hot"],
        "action_modes": ["discrete", "one_hot"],
        "render_modes": ["rgb_array"],
        "env_modes": ["training_priors", "training_task"],
        "render_fps": 4,
    }
    element_to_int = {b"S": 0, b"F": 0, b"H": 1, b"G": 2}

    def __init__(
        self,
        name,
        desc=None,
        is_slippery=False,
        render_mode="rgb_array",
        state_mode="agent_pos",
        action_mode="one_hot",
        env_mode="training_task",
        grid_size=12,
    ):
        assert state_mode is None or state_mode in self.metadata["state_modes"]
        assert action_mode is None or action_mode in self.metadata["action_modes"]
        assert env_mode is None or env_mode in self.metadata["env_modes"]
        self.state_mode = state_mode
        self.action_mode = action_mode
        self.mode = env_mode
        self.grid_size = grid_size
        self.reset(
            seed=None,
            options=None,
            name=name,
            desc=desc,
            is_slippery=is_slippery,
            render_mode=render_mode,
        )
        self.desc = self.unwrapped.desc
        self.observation_space = gym.spaces.Box(
            0, 1, (self.unwrapped.nrow, self.unwrapped.ncol, 4)
        )

    def _change_state_mode(self, state):
        if self.state_mode == "agent_pos":
            return state
        elif self.state_mode == "one_hot":
            return self._one_hot_encode_state(state)
        elif self.state_mode == "flattened_one_hot":
            return self._one_hot_encode_state(state).flatten()
        elif self.state_mode == "rgb_array":
            return self.env.render()

    def reset(
        self,
        seed=None,
        options=None,
        name="FrozenLake-v1",
        desc=None,
        is_slippery=False,
        render_mode="rgb_array",
    )-> tuple[ObsType, float, bool, bool, dict]:
        if not desc:
            env_description = generate_random_map(size=self.grid_size // 2, p=0.8)
            env_description = list(
                map(lambda row: row.replace("F", "S"), env_description)
            )
            # mirror map both vertically and horizontally
            desc = [row[::-1] + row for row in env_description[::-1] + env_description]

        env = gym.make(
            name,
            desc=desc,
            is_slippery=is_slippery,
            render_mode=render_mode,
        )
        super().__init__(env)
        self.s = self.env.reset(seed=seed, options=options)
        state, info = self.s
        return self._change_state_mode(state), info

    def step(self, a):
        self._check_changes()
        next_state, reward, terminated, truncated, info = self.env.step(a)
        if reward == 1 and self.mode == "training_priors":
            reward = 0
        if terminated and reward == 0:
            reward = -1
        self.s = next_state, reward, terminated, truncated, info
        return self._change_state_mode(next_state), reward, terminated, truncated, info

    def render(self):
        x = self.env.render()
        return x

    def _one_hot_encode_state(self, state):
        background_discrete = [
            [self.element_to_int[element] for element in list(row)]
            for row in self.unwrapped.desc
        ]
        background_one_hot = np.eye(3)[background_discrete]
        agent_state = np.zeros((self.unwrapped.nrow, self.unwrapped.ncol, 1))
        row, col = divmod(state, self.unwrapped.ncol)
        agent_state[row, col, 0] = 1
        encoded_state = np.concatenate([background_one_hot, agent_state], axis=-1)
        return encoded_state

    def _check_changes(self):
        row, col = divmod(self.s[0], self.unwrapped.ncol)
        if self.unwrapped.desc[row, col] == b"G":
            self.unwrapped.desc[row, col] = b"F"
