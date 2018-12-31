#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:52:56 2017

@author: farismismar
"""

# Used from: https://keon.io/deep-q-learning/
# Check some more here: https://github.com/leimao/OpenAI_Gym_AI/tree/master/CartPole-v0/Deep_Q-Learning
# https://github.com/keon/deep-q-learning/blob/master/ddqn.py

# Deep Q-learning Agent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.callbacks import History 
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import os

class DQNLearningAgent:
    def __init__(self, seed=0,
                 learning_rate=0.2,
                 discount_factor=1.0,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.91, batch_size=32,
                 state_size=5, action_size=3):

        self.learning_rate = learning_rate          # alpha
        self.discount_factor = discount_factor      # gamma
        self.exploration_rate = exploration_rate    # epsilon
        self.exploration_rate_min = 0.010
        self.exploration_decay_rate = exploration_decay_rate # d
        self.state = None
        self.action = None
        self.seed = seed
        self.state_size = state_size
        self.action_size = action_size
        self.num_actions = action_size
        
        self.model = self._build_model()
        self.target_model = self._build_model()
#        self.update_target_model()
        self.memory = deque(maxlen=2000)
        self.batch_size = batch_size
        
        self._losses = []
        self._history = History()

        self.q = np.random.rand(self.state_size, self.action_size)
        
        # Add a few lines to caputre the seed for reproducibility.
        # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = '0'        
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(seed)        
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

    def _build_model(self):
        # Neural Net for Deep Q learning Model from state_size |S| to action_size |A|
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # was linear?
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
        
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min
    
        return np.zeros(self.state_size) # the action of (nothing done yet).
        
    def act(self, observation, reward=None):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        observation = np.ones(self.state_size,dtype=int) * observation # force for MATLAB implicit conversion.
#        observation = np.array(observation, dtype=int) # force for MATLAB implicit conversion.
        act_values = self.model.predict(observation.reshape(1, self.state_size))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            observation = np.ones(self.state_size,dtype=int) * state # force for MATLAB implicit conversion.
            target = self.model.predict(observation.reshape(1, self.state_size))
            action = np.ones(self.action_size, dtype=int) * action # force for MATLAB conversion
            action = action.reshape(1, self.action_size) 
            if done:
                action = action.astype(int)
                target[0][action] = reward
            else:
                next_observation = np.ones(self.state_size,dtype=int) * next_state # force for MATLAB implicit conversion.
                a = self.model.predict(next_observation.reshape(1, self.state_size))[0]
                t = self.target_model.predict(next_observation.reshape(1, self.state_size))[0]
                action = action.astype(int) # force for MATLAB implicit conversion.
                target[0][action] = reward + self.discount_factor * t[np.argmax(a)]
            #state = np.reshape(state, [1, self.state_size])
            observation = np.ones(self.state_size,dtype=int) * state # force for MATLAB implicit conversion.
            observation = observation.reshape(1, self.state_size)
            #target =  np.ones(self.state_size,dtype=int) * target # force for MATLAB implicit conversion.
            self.model.fit(observation, target, epochs=1, verbose=0, callbacks=[self._history])
            self._losses.append(self._history.history['loss'][0])
            self.q = target  # save the q function.

    def get_losses(self):
        return self._losses

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        return

    def averageQ(self):
        return self.q.mean().mean()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)