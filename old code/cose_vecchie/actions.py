from collections import namedtuple

Experience = namedtuple(
    "Experience", 
    ("state", "action", "next_state", "performed_action_prime")
)

class Action():
    def __init__(self,action_name,layer=None):
        self.name = f"[ACTION {str(action_name)} layer{str(layer)}]"
    
    def __bool__(self):
        return True
    
    def __eq__(self,other):
        return self.name == other.name
    
    def __lt__(self,other):
        return self.name < other.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __str__(self):
        return self.name
    
    def is_fail(self):
        return False    
    
    def unwrap(self):
        return self.name
    
    
class ExploreAction(Action):
    def __init__(self,layer=None):
        super().__init__("EXPLORE",layer)
    
    
class NullAction(Action):
    def __init__(self,layer=None):
        super().__init__("NULL",layer)
        
    def __bool__(self):
        return False
    
    
class FailAction(Action):
    def __init__(self,layer=None):
        super().__init__("FAILED",layer)
    
    def is_fail(self):
        return True
    
    
class TensorAction(Action):
    def __init__(self,state_repr,layer=None): 
        self.repr = state_repr
        super().__init__(str(self.repr),layer)
    
    def unwrap(self):
        return self.repr
    
class TensorDiffAction(Action):
    def __init__(self,state_repr,layer=None): 
        self.repr = self.sparse_delta_repr(state_repr)
        super().__init__(str(self.repr),layer)
    
    def sparse_delta_repr(self, state_repr):
        element_list = []
        for x,row in enumerate(state_repr):
            for y,element in enumerate(row):
                if element: element_list.append((x,y,element))
        (x1,y1,v1),(x2,y2,v2) = element_list
        deltax,deltay = (x2-x1)*(v2-v1)/2,(y2-y1)*(v2-v1)/2
        direction = ""
        if deltax>0: direction+="down"
        if deltax<0: direction+="up"
        if deltay>0: direction+="right"
        if deltay<0: direction+="left"
        return direction
    
    def unwrap(self):
        return self.repr
