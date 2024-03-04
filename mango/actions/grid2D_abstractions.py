from dataclasses import dataclass
import sys 
sys.path.append('..')
from enum import IntEnum
import random
from typing import ClassVar, Optional
import numpy as np
from mango.protocols import ObsType, ActType, Transition
from  mango import spaces
import numpy.typing as npt
from dataclasses import field
from mango.utils import is_traversable
import torch
from copy import deepcopy

class Actions(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    TASK = 4

    def to_delta(self) -> tuple[int, int]:
        return {
            Actions.LEFT: (0, -1),
            Actions.DOWN: (1, 0),
            Actions.RIGHT: (0, 1),
            Actions.UP: (-1, 0),
            Actions.TASK: (0, 0),
        }[self]
        
@dataclass(eq=False, slots=True, repr=True)
class GridMovement:
    cell_shape: tuple[int, int]
    target_delta: tuple[int, int]
    edges: npt.NDArray[np.int32] = field(default=None)
    
    action_space: ClassVar = spaces.Discrete(len(Actions))

    def abstract(self, obs: ObsType) -> tuple[int, int]:
        # assume obs has shape (C, Y, X) and the agent pos in 1hot encoded in channel 0
        idx = int(np.argmax(obs[0, 1:-1, 1:-1]))
        y, x = divmod(idx, obs[0, 1:-1, 1:-1].shape[1])
        return y // self.cell_shape[0], x // self.cell_shape[1]
    
    def abstract_gift(self, obs: ObsType) -> tuple[int, int]:
        # assume obs has shape (C, Y, X) and the gift pos in 1hot encoded in channel 1
        idx = int(np.argmax(obs[2, 1:-1, 1:-1]))
        y, x = divmod(idx, obs[2, 1:-1, 1:-1].shape[1])
        return y // self.cell_shape[0], x // self.cell_shape[1]
    
    def abstract_edges(self, obs):
        # Unframed observation
        unframed_obs = deepcopy(obs)
        # Initialize expanded_matrix to store the result
        expanded_matrix = np.zeros((3, (unframed_obs.shape[1]//self.cell_shape[0])*2-1, (unframed_obs.shape[2]//self.cell_shape[1])*2-1), dtype=np.int32)
        for i in range(unframed_obs.shape[1]//self.cell_shape[0]):
            for j in range(unframed_obs.shape[2]//self.cell_shape[1]):
                # Extract 2x2 sub-grid from framed_obs
                obs_check = deepcopy(obs[:, i*self.cell_shape[0]:(i+1)*self.cell_shape[0]+2, j*self.cell_shape[1]:(j+1)*self.cell_shape[1]+2])
                obs_check[1,[0,0,-1,-1],[0,-1,0,-1]]=1
                starting_point = np.where(obs_check[1, 1:-1, 1:-1] == 0)
                if starting_point[0].size == 0:
                    expanded_matrix[1, 2*i, 2*j+1] = 0
                else:
                    # Process lakes channel for horizontal connectivity
                    if j!=unframed_obs.shape[2]//self.cell_shape[1]-1:
                        if torch.any(obs_check[1, :, -1] == 0):
                            index_of_first_zero = np.where(obs_check[1, :, -1] == 0)[0][0]
                            expanded_matrix[1, 2*i, 2*j+1] = is_traversable(obs_check, (starting_point[0][0]+1,starting_point[1][0]+1), (index_of_first_zero, self.cell_shape[0]+1))
                        else:
                            expanded_matrix[1, 2*i, 2*j+1] = 0
                    
                    # Process lakes channel for vertical connectivity
                    if i!=unframed_obs.shape[1]//self.cell_shape[0]-1:
                        if torch.any(obs_check[1, -1, :] == 0):
                            index_of_first_zero = np.where(obs_check[1, -1, :] == 0)[0][0]
                            expanded_matrix[1, 2*i+1, 2*j] = is_traversable(obs_check, (starting_point[0][0]+1,starting_point[1][0]+1), (self.cell_shape[0]+1, index_of_first_zero))
                        else:
                            expanded_matrix[1, 2*i+1, 2*j] = 0
        return expanded_matrix
    
    def set_up_edges(self, obs):
        self.edges = self.abstract_edges(obs)
    
    def full_abstraction(self, obs: ObsType) -> npt.NDArray[np.int32]:
        agent_pos = self.abstract(obs)
        gift_pos = self.abstract_gift(obs)
        if self.edges is None:
            self.set_up_edges(obs)
        expanded_matrix = deepcopy(self.edges)
        expanded_matrix[0,agent_pos[0]*2,agent_pos[1]*2]=1
        expanded_matrix[2,gift_pos[0]*2,gift_pos[1]*2]=1
        return expanded_matrix
        

    def beta(self, trajectory: list[Transition]) -> bool:
        start = self.abstract(trajectory[0].start_obs)
        end = self.abstract(trajectory[-1].next_obs)
        return start != end

    def reward(self, trajectory: list[Transition]) -> float:
        start = self.abstract(trajectory[0].start_obs)
        end = self.abstract(trajectory[-1].next_obs)
        target = tuple(s + d for s, d in zip(start, self.target_delta))

        if start == target:
            return -0.0  # no action, no reward

        if end == target:
            return 1.0
        else:
            return -1.0

    def mask(self, obs: ObsType) -> ObsType:
        masked_obs = self.full_abstraction(obs)
        padded_shape = (masked_obs.shape[0], masked_obs.shape[1] + 2, masked_obs.shape[2] + 2)
        padded_obs = np.zeros_like(masked_obs, shape=padded_shape)
        padded_obs[:, 1:-1, 1:-1] = masked_obs
        padded_obs[1, :, [0,-1]] = 1
        padded_obs[1, [0,-1], :] = 1

        y, x = self.abstract(obs)
        big_x, big_y = x//2, y//2
        y_min_padd = big_y * 2*2
        y_max_padd = (big_y+1) * 2*2+1
        x_min_padd = big_x * 2*2
        x_max_padd = (big_x+1) * 2*2+1
        masked_obs = padded_obs[:, y_min_padd:y_max_padd, x_min_padd:x_max_padd]
        return masked_obs


class MoveLeft(GridMovement):
    def __init__(self, cell_shape: tuple[int, int]):
        super().__init__(cell_shape, d_y=0, d_x=-1)


class MoveDown(GridMovement):
    def __init__(self, cell_shape: tuple[int, int]):
        super().__init__(cell_shape, d_y=1, d_x=0)


class MoveRight(GridMovement):
    def __init__(self, cell_shape: tuple[int, int]):
        super().__init__(cell_shape, d_y=0, d_x=1)


class MoveUp(GridMovement):
    def __init__(self, cell_shape: tuple[int, int]):
        super().__init__(cell_shape, d_y=-1, d_x=0)
        
        


