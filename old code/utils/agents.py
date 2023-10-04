
class MangoAgent():
    @property
    def task_policies(self)->list[Policy[State, Action]]:



    def step(self, at_layer=None):
        layer = at_layer if at_layer is not None else self.mango.n_layers-1

        start_state = self.mango.current_state(layer)
        action_idx = self.policy_list[layer](start_state)
        if action_idx != 0:
            reward, option_reward = self.mango.handle_option(layer, action_idx-1) 
        else:
            reward, option_reward = self.handle_terminal_option(layer-1)
        end_state = self.mango.current_state(layer)
        self.replay_memory_list[layer].push(start_state, action_idx, end_state, reward)
        
        return reward if at_layer is None else reward, action_idx 
    
    def handle_terminal_option(self, layer):
        if layer < 0: 
            return 0
        
        cumulative_reward = 0
        while True:
            reward, action_idx = self.step(layer)
            cumulative_reward += reward
            if action_idx == 0:
                break
        return cumulative_reward
    
    def train(self, layer=None, cycles=1):
        for layer, (policy, memory) in enumerate(zip(self.policy_list,self.replay_memory_list)):     
            if memory.can_provide_sample:
                for i in range(cycles):
                    start_state_list, actions_idx_list, end_state_list, reward_list = memory.sample()
                    policy.train(start_state_list, actions_idx_list, end_state_list, reward_list)
    




        
            