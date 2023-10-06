from actions import *
from states import *


class StateAbstractor():    
    def __call__(self,state):
        return self.abstract(state)
    
    def abstract(self, state):
        """Compute the abstraction of a given state."""
        # TO DO: write the abstarction function
        state_prime = state
        return state_prime
    
    def get_action_prime(self,old_state_prime,new_state_prime,layer=None):
        """Compute the abstract action associated with an abstract state transition."""
        if old_state_prime == new_state_prime:
            return NullAction(layer)
        else:
            return TensorDiffAction(new_state_prime-old_state_prime,layer)
    
    
    
class AbstractionBuffer():
    def __init__(self,abstractor, layer=None):
        self.abstractor = abstractor
        self.layer = layer
        self.reset()
    
    def __call__(self,state):
        return self.push(state)
    
    def get_current_state(self):
        """returns the tuple (state, state')."""
        return self.current_state, self.current_state_prime
    
    def push(self, new_state):
        """forwards the state, computes state' and returns the performed action'."""
        old_state, old_state_prime = self.get_current_state()
        
        new_state_prime = self.abstractor(new_state)
        self.current_state, self.current_state_prime = new_state, new_state_prime
        
        if old_state_prime is None or new_state_prime is None: 
            return NullAction()
        
        action_prime = self.abstractor.get_action_prime(old_state_prime,new_state_prime,self.layer)   
        return action_prime

    def reset(self):
        self.current_state = None
        self.current_state_prime = None
        
    
    
        
