from logging import raiseExceptions
import random
import copy
from collections import namedtuple,OrderedDict

import matplotlib.pyplot as plt

import numpy as np
import torch

from actions import *
from states import *

class ReplayMemory():
    """Class keeping track of experiences, divided according to origin (command)"""
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = {}
        self.push_count = {}
    
    def push(self, key, experience):
        """add experience collected following command idx"""
        if not key in self.memory:
            self.memory[key]=[]
            self.push_count[key]=0
            
        if len(self.memory[key]) < self.capacity:
            self.memory[key].append(experience)
        else:
            self.memory[key][self.push_count[key] % self.capacity] = experience
        self.push_count[key] +=1
    
    def sample(self, quantity=None ,key_sample_weights=None):
        """sample from memories according to frequency dictionary {idx: sample_prob}"""
        if quantity is None: quantity = self.batch_size
        if key_sample_weights is None: key_sample_weights={i:1 for i in self.memory.keys()}
        
        keys, weights = zip(*key_sample_weights.items())
        probs = np.array(weights, dtype=float) / float(np.sum(weights))
        
        keys, counts = np.unique(np.random.choice(keys, quantity, p=probs), return_counts=True)

        sample = np.concatenate([self.sample_local(key, count) for key,count in zip(keys,counts)])
        return sample
    
    def sample_local(self,key,quantity=None):
        """sample from single memory"""
        if quantity is None: quantity = self.batch_size
        return random.choices(self.memory[key], k=quantity)
        
    def can_provide_sample(self, size=None):
        if size is None: size = self.batch_size
        return self.size() >= size
    
    def size(self, key=None):
        if key is None: 
            return np.sum([len(mem) for idx, mem in self.memory.items()])
        return len(self.memory.get(key,[]))
    

class PolicyStack(OrderedDict):
    """Create a class able to save the policies and access to them when needed.
    This class also keeps track of the replay memory"""
    
    def __init__(self, policy, layer=None, capacity = 1000, batch_size = 64):
        super(PolicyStack, self).__init__()
        self.current_comand = ExploreAction(layer)
        self.add_new_policy(self.current_comand, policy)
        self.memory = ReplayMemory(capacity, batch_size)
        self.action_range = 1
        
    def set_action_space(self, action_space):
        self.action_to_idx = {a:idx for idx,a in enumerate(action_space)}
        self.idx_to_action = action_space
        for key, policy in self.items():
            policy.set_action_range(len(action_space))
        
    def set_comand(self,action_prime):
        """Sets the current policy."""
        if not self.__contains__(action_prime):
            raise Exception("Unknown command: cannot reproduce never seen before behaviour")
        self.current_comand = action_prime
    
    def get_policy(self):
        """Return the current policy"""
        return self.__getitem__(self.current_comand)
        
    def get_action(self,state):
        """Returns the action suggested by the current policy."""
        if self.current_comand == next(iter(self.keys())):
            return random.choice(self.idx_to_action)
        current_policy = self.get_policy()
        return self.idx_to_action[current_policy(state)]
        
    def feedback(self,episode,performed_action_prime):
        """Handle the feedback from mango."""
        # update memory of policystack 
        self.update_policy_memory(episode,performed_action_prime)
        
        # set training parameters
        if not performed_action_prime:
            # if nothing happens, no training?
            train_focus = None
            train_cycles = 1
            
        elif performed_action_prime.is_fail():
            # if failing to reproduce comand, then focus train on current command
            train_focus = [self.current_comand]
            train_cycles = 10
            
        elif performed_action_prime == self.current_comand:
            # if right action, light unfocused training
            train_focus = None
            train_cycles = 10
            
        elif performed_action_prime != self.current_comand:
            # if wrong action, then focus train on current command and on mistaken action
            train_focus = [self.current_comand,performed_action_prime]
            train_cycles = 20
            
        self.train_current_policy(focus=train_focus, cycles=train_cycles)            

    def add_new_policy(self,action_prime,policy=None):
        """Adds a new policy associated to action prime. 
        if no policy is given, splits the current active one"""
        if policy is None:
            policy = self.get_policy().split()
        return self.__setitem__(action_prime,policy)
    
    def update_policy_memory(self,episode,performed_action_prime):
        for experience in episode:
            self.memory.push(self.current_comand,experience)
            self.memory.push(performed_action_prime,experience)
        
    def train_current_policy(self, focus=None, cycles=1):
        """sample from replay memory and invoke train method for the current policy"""
        if not self.memory.can_provide_sample():
            return
        
        #print(f"training {self.current_comand}")
        for i in range(cycles):
            key_sample_weights = None if focus is None else {key:self.memory.size(key) for key in focus}
            sampled_experiences = self.memory.sample(key_sample_weights=key_sample_weights)
            states, actions, new_states, performed_action_primes = zip(*sampled_experiences)
            rewards = [self.get_reward(action_prime) for action_prime in performed_action_primes]
            actions_idx = [self.action_to_idx[action] for action in actions]

            self.get_policy().train(states, actions_idx, new_states, rewards)
        
    def get_reward(self, action_prime, target=None):
        if target is None: target = self.current_comand
        if not action_prime: return -1
        if action_prime.is_fail(): return -5
        if action_prime==target: return 10
        else: return -10
        

class QnetPolicy():
    """
    the policy will be not only the model to predict the next action but also 
    it will contain the memory and its trainiing process
    """
    def __init__(self, Q_net):
        self.Q_net = Q_net        
        self.set_optimizer()
        self.losses = []
        self.refresh_target()
    
    def refresh_target(self, steps=10):
        self.target_Q_net = self.Q_net
        self.refresh_timer = steps
    
    def __call__(self, state, epsilon=0.0):
        """epsilon greedy strategy"""
        if np.random.sample()<epsilon:
            action_prob = torch.zeros(self.action_range)
        else:
            state = state.unwrap().unsqueeze(0)
            action_prob = self.Q_net(state)[0,:self.action_range]
        #action = torch.distributions.Categorical(logits=action_prob).sample()
        action = torch.argmax(action_prob)
        return action
    
    def split(self):
        """make an initialazed copy"""
        return QnetPolicy(copy.deepcopy(self.Q_net))
    
    def set_action_range(self, n):
        self.action_range = n
        
    def set_optimizer(self):
        """initiate the optimizer for the policy"""
        self.optimizer = torch.optim.RMSprop(params=self.Q_net.parameters(), lr=0.01)
        
    def train(self, states, actions_idx, new_states, rewards, gamma=0.99):
        """train policy for one step"""
        if self.refresh_timer == 0:
            self.refresh_target()
            
        # unwrap
        states = torch.stack([s.unwrap() for s in states])
        new_states = torch.stack([s.unwrap() for s in new_states])
        actions_idx = torch.tensor(actions_idx).unsqueeze(1)
        rewards = torch.tensor(rewards)
        
        # get q_values
        current_q_values = self.Q_net(states)[:,:self.action_range]
        current_q_values = torch.gather(current_q_values,1,actions_idx)[:,0]
        target_q_values = self.target_Q_net(new_states)[:,:self.action_range]
        target_q_values = torch.max(target_q_values,dim=1)[0]* gamma + rewards
        
        # evaluate loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(current_q_values, target_q_values)
        self.losses.append(torch.mean(loss).item())
        
        #train parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def plot(self,*args):
        """plot the loss and the mean reward done by the policy"""
        plt.plot(self.losses) 

        
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).squeeze(1).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).squeeze(1).max(dim=1)[0].detach()
        return values

        
