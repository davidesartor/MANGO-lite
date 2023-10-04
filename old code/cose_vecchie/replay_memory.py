import random
from collections import namedtuple
import torch

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {}
        self.push_count = {}
    
    def cumulative_memory(self):
        return sum([len(self.memory[k]) for k in self.memory.keys()])
    
    def push(self, experience, action):
        self.memory.setdefault(action,[])
        self.push_count.setdefault(action,0)
        if len(self.memory[action]) < self.capacity:
            self.memory[action].append(experience)
        else:
            self.memory[action][self.push_count % self.capacity] = experience
        self.push_count[action] +=1
    
    def sample(self, batch_size, action):
        return random.sample(self.memory[action], batch_size)
    
    def can_provide_sample(self, batch_size, action):
        return len(self.memory[action]) >= batch_size
    
def extract_tensors(Experience,experiences):
    
    batch = Experience(*zip(*experiences))
    
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    
    return (t1, t2, t3, t4)
