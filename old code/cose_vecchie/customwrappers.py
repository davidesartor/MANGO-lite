import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn

import numpy as np
import matplotlib.pyplot as plt

import gym
import minigrid

from gym.core import ObservationWrapper, Wrapper

class FreeMovementEnvWrapper(Wrapper):
    """
    Wrapper for modify environment behavior (rewards, actionspace, etc)
    change action space to allow:
        0: right
        1: down
        2: left
        3: up
        4: pick-up
        5: drop
        6: toggle
    """

    def __init__(self, env):
        super().__init__(env)
        

    def step(self, action):
        # if action is movement command
        if action <= 3:
            # rotate agent untill alignment
            while self.unwrapped.agent_dir != action:
                obs, reward, terminated, truncated = self.env.step(0)
            # take step
            obs, reward, terminated, truncated = self.env.step(2)
            
        # if not movement comand, action_internal = action-1
        else:
            obs, reward, terminated, truncated = self.env.step(action-1)

        return obs, reward, terminated, truncated

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    
class OneHotObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of agent view as observation.
    """

    def __init__(self, env, max_objects=32, max_state_objects=10):
        super().__init__(env)
        self.max_objects = max_objects
        self.max_state_objects = max_state_objects
        # dictionary of objects found (key = str((OBJECT_IDX, COLOR_IDX, STATE))) : item = onehotencoding)
        self.object_dict = {}
        self.item_dict = {}
        self.abstract_actions = {}
        self.old_abstract_state = None
        
        # create new img space ( box of shape (row, cols, maxobjects) )
        obs_shape = env.observation_space["image"].shape
        new_image_space = gym.core.spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], max_objects), dtype="uint8"
        )
        
        # override img field in the obs dict with new img space
        self.observation_space = gym.core.spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )
        self.height,self.width,self.depth=self.observation_space["image"].shape
        
    def observation(self, obs):
        img = obs["image"]
        del obs["direction"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")
        state_space = np.zeros(self.max_state_objects)
        # for each position in the img
        for i,row in enumerate(img):
            for j,item in enumerate(row):
                # check if already seen, if not add it to the dictionary and assign a new encoding dimension
                self.object_dict.setdefault(str(item),len(self.object_dict))
                
                #check if it's not the agent, the walls or a clean grid
                if (item==np.array([4,4,0])).all():
                    item=np.array([1,0,0])
                if item[0]==10 or (item==np.array([1,0,0])).all() or (item==np.array([2,5,0])).all():
                    None
                else:
                    self.item_dict.setdefault(tuple(item),len(self.item_dict))
                    # save state space
                    state_space[self.item_dict[tuple(item)]] = 1
                # save one-hot encoding for this position
                out[i,j,self.object_dict[str(item)]] = 1

        return {**obs, "image": out, "state": state_space}
    
    def _reset(self):
        reset = self.env.reset()[0]
        del reset["direction"]
        img = reset["image"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")
        state_space = np.zeros(self.max_state_objects)
        # for each position in the img
        for i,row in enumerate(img):
            for j,item in enumerate(row):
                # check if already seen, if not add it to the dictionary and assign a new encoding dimension
                self.object_dict.setdefault(str(item),len(self.object_dict))
                
                #check if it's not the agent, the walls or a clean grid
                if (item==np.array([4,4,0])).all():
                    item=np.array([1,0,0])
                if item[0]==10 or (item==np.array([1,0,0])).all() or (item==np.array([2,5,0])).all():
                    None
                else:
                    self.item_dict.setdefault(tuple(item),len(self.item_dict))
                    # save state space
                    state_space[self.item_dict[tuple(item)]] = 1
                # save one-hot encoding for this position
                out[i,j,self.object_dict[str(item)]] = 1
                reset
        return {**reset, "image": out, "state": state_space}
    
    def reset(self):
        reset = self._reset()
        self.old_abstract_state = reset["state"]
        return reset
    
    def _collect_abstract_actions(self,abstract_state):
        action = abstract_state - self.old_abstract_state
        self.old_abstract_state = abstract_state
        if (action!=np.zeros(len(action))).any() and np.array([(action!=i).any() for i in self.abstract_actions.values()]).all():
            self.abstract_actions[len(self.abstract_actions)+1] = action
    
    def _reward(self):
        return 1 - 0.9 * (self.env.step_count / self.env.max_steps)
        
    def step(self, action, abstract_action=None):
        # modification of the output of step function in order to take the abstract state easily
        obs, reward, terminated, truncated, _ = self.env.step(action)
        reward = 0 
        obs = self.observation(obs)
        image = obs["image"]
        abstract_state = obs["state"]
        if not abstract_action:
            None
        elif ((abstract_state-self.old_abstract_state) == self.abstract_actions[abstract_action]).all():
            reward =  self._reward()
        elif (abstract_state == self.old_abstract_state).all():
            reward = 0
        elif ((abstract_state-self.old_abstract_state) != self.abstract_actions[abstract_action]).any() :
            reward =  -1
        self._collect_abstract_actions(abstract_state)
        return image, abstract_state, torch.tensor([reward]), terminated, truncated
    
    def num_actions_available(self):
        return self.env.action_space.n

        
        

