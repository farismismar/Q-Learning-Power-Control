#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:34:27 2018

@author: farismismar
"""
import random
import numpy as np

# Following from
# https://github.com/YuriyGuts/cartpole-q-learning/blob/master/cartpole.py
# This is not a Deep Q Learning Agent

class QLearningAgent:
    def __init__(self, seed,
                 learning_rate=0.2,
                 discount_factor=1.0,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.99, batch_size=32,
                 state_size=3, action_size=5):
        # Episode 458 is the first episode for epsilon min
        self.learning_rate = learning_rate          # alpha
        self.discount_factor = discount_factor      # gamma
        self.exploration_rate = exploration_rate    # epsilon
        self.exploration_rate_min = 0.010
        self.exploration_decay_rate = exploration_decay_rate # d
        self.state = np.zeros(state_size, dtype=int)
        self.action = 0 #np.zeros(action_size, dtype=int)
        self.seed = seed
        self.batch_size = batch_size # dummy variable -- does nothing
        self.num_actions = action_size
        self.state_size = state_size
        
        self.memory = []
        
        # Add a few lines to caputre the seed for reproducibility.
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Create a clean Q-Table.
        self.action_size = action_size
        self.q = np.zeros(shape=(state_size, self.num_actions))
        
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min
    
        # Get the action for the initial state.
        self.state = observation + np.zeros(self.state_size, dtype=int)
        state = self.state[0]
        return np.argmax(self.q[state])
        
    def act(self, observation, reward):
        next_state = observation[0]
        state = self.state
        if isinstance(state, np.ndarray):
            state = state[0]
        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)
        if enable_exploration:
            next_action = np.random.randint(0, self.num_actions)
        else:
            next_action = np.argmax(self.q[next_state])
        
        # Learn: update Q-Table based on current reward and future action.
        self.q[state, self.action] += self.learning_rate * \
            (reward + self.discount_factor * max(self.q[next_state, :]) - self.q[state, self.action])
    
        self.state = next_state
        self.action = next_action
        return next_action

    def remember(self, prev_observation, action, reward, observation, done):
        return # this is a dummy function for compatibility
    
    def get_losses(self):
        return []
    
    def update_target_model(self):
        return