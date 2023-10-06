import warnings
warnings.filterwarnings("ignore")

import math
import torch
import numpy as np
from collections import namedtuple
import torch.optim as optim
from itertools import count
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gym
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from customwrappers import FreeMovementEnvWrapper, OneHotObsWrapper

from agent import EpsilonGreedyStrategy, Agent, Abstract_agent
from replay_memory import ReplayMemory,extract_tensors
from models import QValues,DQN_CNN,QTable
from plot import plot

batch_size = 50
gamma = 0.99
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 50 

memory_size = 100000
lr = 0.01
num_episodes = 1000

Experience = namedtuple(
    "Experience", 
    ("state", "action", "next_state", "reward")
)

device = torch.device("cuda" if torch .cuda.is_available() else "cpu")

env_base = gym.make('MiniGrid-DoorKey-5x5-v0', render_mode="rgb_array",highlight = False)
env_obs = FullyObsWrapper(env_base)
env_onehot = OneHotObsWrapper(env_obs,max_objects=16)

env_onehot.reset()
img = env_onehot.render()
plt.imshow(img)
print("environment set up")

strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, env_onehot.num_actions_available(), device)
abstract_agent = Abstract_agent(strategy, device)
memory = ReplayMemory(memory_size)

policy_net = DQN_CNN(env_onehot.height,env_onehot.width,env_onehot.depth,env_onehot.num_actions_available()).to(device)
target_net = DQN_CNN(env_onehot.height,env_onehot.width,env_onehot.depth,env_onehot.num_actions_available()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer_dict = {}


import numpy as np
q_table = QTable()

alpha_abstract = 0.1
gamma_abstract = 0.6
epsilon_abstract = 0.1

plt.ion()
fig, f = plt.subplots()

episode_reward=[]
losses_tot={}


for episode in range(0,num_episodes):
    obs = env_onehot.reset()
    image = obs["image"].copy()
    abstract_state = obs["state"].copy()
    
    img = env_onehot.render()
    f.imshow(img)
    plt.pause(.1)

    
    tot_reward = 0
    for abstract_step in count():
        abstract_action = abstract_agent.select_action(abstract_state,q_table)
        
        optimizer = optimizer_dict.setdefault(abstract_action,optim.RMSprop(params=policy_net.parameters(), lr=0.001))
        for ministep in count():
            action = agent.select_action(torch.tensor(image).permute(2,0,1).float().unsqueeze(0), policy_net, abstract_action)
            state = image.copy()
            old_abstract_state = abstract_state.copy()
            image, abstract_state, reward, terminated, truncated = env_onehot.step(action,abstract_action)
            abstract_agent.num_actions = len(env_onehot.abstract_actions)
            tot_reward += reward
            
            img = env_onehot.render()
            f.imshow(img)
            plt.show() 
            plt.pause(.5)
 


            memory.push(Experience(torch.tensor(image).permute(2,0,1).float().unsqueeze(0), action, torch.tensor(image).permute(2,0,1).float().unsqueeze(0), reward),abstract_action)
            losses = []
            if memory.can_provide_sample(batch_size,abstract_action):
                experiences = memory.sample(batch_size,abstract_action)
                states, actions, rewards, next_states = extract_tensors(Experience,experiences)
                current_q_values = QValues.get_current(policy_net, states, actions,abstract_action)
                next_q_values = QValues.get_next(target_net, next_states,abstract_action)
                target_q_values = (next_q_values * gamma) + rewards
                criterion = nn.SmoothL1Loss()
                
                loss = criterion(current_q_values, target_q_values.unsqueeze(1))
                losses.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if ministep%10==0:
                print(ministep)
            if terminated or (ministep+1)%100==0:
                if not abstract_action:
                    abstract_action=-1
                else:
                    losses_tot.setdefault(abstract_action,[])
                    losses_tot[abstract_action].append(np.mean(losses))
                    print(losses_tot)
                break
            
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        abstract_reward = 0
        if abstract_state==np.zeros(env_onehot.max_objects):
            abstract_reward = 1
        
        old_value = q_table.take(old_abstract_state, abstract_action)
        q_table.take(abstract_state, abstract_action)
        next_max = q_table.val_max(abstract_state)
        
        new_value = (1 - alpha_abstract) * old_value + alpha_abstract * (abstract_reward + gamma_abstract * next_max)
        q_table.q_table.loc[tuple(old_abstract_state), abstract_action] = new_value

    episode_reward.append(tot_reward)
    if episode % 100 == 0:
        print(f"Episode: {episode}")
        for i in q_table.q_table.columns:
            plot(losses_tot[i],10)