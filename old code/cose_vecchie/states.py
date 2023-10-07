import torch

class State():
    def __init__(self,state_repr,layer=None): 
        self.name = f"[STATE {str(state_repr)} layer{str(layer)}]"
        self.repr = state_repr
        
    def __bool__(self):
        return True
    
    def __add__(self,other):
        return self.repr + other.repr 
        
    def __sub__(self,other):
        return self.repr - other.repr 
        
    def __mul__(self,other):
        return self.repr * other.repr 
    
    def __eq__(self,other):
        return self.repr == other.repr
    
    def __hash__(self):
        return hash(self.name)
    
    def __str__(self):
        return self.name   
    
    def unwrap(self):
        return self.name
    
class TensorState(State):
    def __init__(self,state_repr,layer=None): 
        self.repr = state_repr
        self.name = f"[STATE {str(self.sparse_delta_repr())} layer{str(layer)}]"
    
    def sparse_delta_repr(self):
        element_list = []
        for x,row in enumerate(self.repr):
            for y,element in enumerate(row):
                if element: element_list.append((f"({x},{y}):{element})"))    
        return element_list
    
    def __eq__(self,other):
        return torch.equal(self.repr, other.repr)
    
    def unwrap(self):
        return self.repr