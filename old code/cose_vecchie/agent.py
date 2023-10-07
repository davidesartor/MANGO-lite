import math
import random
import torch

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
    
class Agent():
    def __init__(self, strategy, num_actions, device): #device that we use in pytorch for tensor calculation CPU or GPU
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.dict_agents = {}
    
    def select_action(self, state, policy_net, abstract_action):
        if not abstract_action:
            action = random.randrange(self.num_actions) #explore
            return torch.tensor([action]).to(self.device)
        current_step = self.dict_agents.setdefault(abstract_action,0)
        rate = self.strategy.get_exploration_rate(current_step)
        self.dict_agents[abstract_action] += 1
        
        if rate > random.random():
            action = random.randrange(self.num_actions) #explore
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state,abstract_action).argmax(dim=1).to(self.device) #exploit
            
class Abstract_agent():
    def __init__(self, strategy,device, num_actions = 0): 
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
    
    def select_action(self, abstract_state, policy_net):  
        abstract_state = tuple(abstract_state)
        rate = self.strategy.get_exploration_rate(self.current_step)
        if self.num_actions <= 1:
            return None
        if rate > random.random():
            return None
        else:
            if rate > random.random():
                abstract_action = random.randrange(self.num_actions,1) #explore
                self.current_step +=1
                return torch.tensor([abstract_action]).to(self.device)
            else:
                with torch.no_grad():
                    self.current_step +=1
                    return policy_net.q_table.loc[abstract_state].idxmax(axis = "columns").to(self.device) #exploit
                
        