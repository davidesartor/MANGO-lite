import numpy as np
from actions import *
from states import *
from abstraction import AbstractionBuffer,StateAbstractor


class Mango():
    def __init__(self, slave_mango, policy_stack, abstractor, layer=None):
        self.slave_mango = slave_mango
        self.policy_stack = policy_stack
        self.layer = layer
        
        self.abstraction_buffer = AbstractionBuffer(abstractor,layer)
        self.reset_buffer()
    
    def __call__(self,action_prime):
        return self.handle_abstract_action(action_prime)
    
    def get_current_state(self):
        return self.abstraction_buffer.get_current_state()
    
    def learned_comands(self):
        return list(self.policy_stack.keys())
        
    def allow_more_actions(self):
        """decide if more actions can be performed."""        
        expected_max_trials = 10
        return np.random.random() > 1/expected_max_trials
    
    def handle_abstract_action(self, action_prime):
        """handle the sequencing of actions needed to perform a given abstract action."""        
        # give abstract action comand to policystack
        self.policy_stack.set_comand(action_prime)
        self.policy_stack.set_action_space(self.slave_mango.learned_comands())
        performed_action_prime = NullAction(self.layer)
        episode = []
        
        while not performed_action_prime:
            # get the action
            state,state_prime = self.abstraction_buffer.get_current_state()
            action = self.policy_stack.get_action(state)
            
            # downstream handling of action
            new_state = self.slave_mango(action)
            
            # forward the state to the abstractor and get the abstract action was performed
            performed_action_prime = self.abstraction_buffer(new_state)
            
            if performed_action_prime:
                if performed_action_prime not in self.learned_comands():
                    print(f"discovered new action {performed_action_prime}")
                    # if never seen this acton, split current policy and extend master's action range
                    self.policy_stack.add_new_policy(performed_action_prime)
            else:
                if not self.allow_more_actions():
                    performed_action_prime = FailAction(self.layer)
                
            # give feedback to policystack
            episode.append(Experience(state,action,new_state,performed_action_prime))
            
        self.policy_stack.feedback(episode,performed_action_prime)
        return state_prime
    
    def reset_buffer(self):
        self.abstraction_buffer.reset()
        action_prime = self.abstraction_buffer.push(self.slave_mango.get_current_state()[1])
    
    def reset_env(self):
        self.slave_mango.reset_env()
        self.reset_buffer()
    
    
class MangoEnvironment():
    """Wrap an environment to make it beheave like a mango"""
    def __init__(self, env, state_wrapper=State, action_wrapper=Action):
        self.env = env 
        self.reset_env()
        self.state_wrapper = state_wrapper
        self.action_wrapper = action_wrapper
    
    def __call__(self,action):
        self.state, *_  = self.env.step(action.unwrap())
        return self.state_wrapper(self.state)
    
    def learned_comands(self):
        return [self.action_wrapper(a) for a in self.env.actionspace]
    
    def get_current_state(self):
        return None, self.state_wrapper(self.state)
    
    def reset_buffer(self):
        return 
    
    def reset_env(self):
        self.state = self.env.reset()
        return self.state

        
        
       