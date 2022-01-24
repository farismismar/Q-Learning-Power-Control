#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:26:31 2017

@author: farismismar
"""

import numpy as np 

# An attempt to follow
# https://github.com/openai/gym/blob/master/gym/envs

class SINR_environment:
    
    def __init__(self, initial_value, target_value, random_state, state_size=3, action_size=5):
        np.random.seed(random_state)
  
        self.initial_value = initial_value # initial SINR
        self.target_value = target_value   # Desired target from outer loop PC

        self.state_size = state_size
        self.action_size = action_size
        self.observation_space = np.arange(action_size) # actions are [0, -3, -1, 1, 3]
#        self.action_space = np.arange(action_size) # from 0 (to action_size - 1)
        self.score = 0.                                  # current SINR value
        self.random_state = random_state
 #       self.last_reward = 0.
        self.observation = 0 + np.zeros(state_size, dtype=int)         # current state: No attempt on SINR improvement yet
 #       self.last_action = None
        self.reset()
        self.seed(random_state)
        self.iter_count = 0.
    
    def seed(self, random_state):
        np.random.seed(self.random_state)
        
    def reset(self):
        self.score = 0.
        self.iter_count = 0.
#        self.last_reward = 0.
        self.observation = 0
#        self.last_action = None
        return np.zeros(self.state_size, dtype=int)  # all states/actions are scalars repeated n times as a vector

    def close(self):
        self.reset()
    
    def step(self, action):
        reward = 0.
        
        # Check if action is integer
        if isinstance(action, np.ndarray):
            action = action[0]
        
         # actions: 0 nothing
         # actions: 1, 2 this is a power down
         #          : 3, 4 this is a power up

        # The next states based on the action
        if action == 0:
            self.observation = 0 + np.zeros(self.state_size, dtype=int) # do nothing
            reward = -100 # Do not come here again.
        elif action == 1 or action == 2:
            self.observation = 2 + np.zeros(self.state_size, dtype=int) # SINR has reduced (either by -1 or -3), state = 2
            reward = -1
            #print(action, self.observation, reward)
        elif action == 3 or action == 4:
            self.observation = 1 + np.zeros(self.state_size, dtype=int) # SINR has increased (either by 1 or 3), state = 1
            reward = 1
           # print(action, self.observation, reward)
        else:
            self.observation = 0 + np.zeros(self.state_size, dtype=int) # do nothing
            reward = -100

        #print('Reward is: {}'.format(reward))
        
       # done = (self.score >= self.target_value)
        self.iter_count += 1
        return self.observation, reward, None, self.iter_count