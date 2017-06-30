#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:20:05 2017

@author: farismismar
"""

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib import rc

from environment import SINR_environment
from DQNAgent import DQNAgent

import os
os.chdir('/Users/farismismar/Desktop/E_Projects/UT Austin Ph.D. EE/Papers/Deep Reinforcement Learning in RF Optimization in mmWave Propagation/')

seed=1

def cost231(distance, f, h_R, h_B):
    C = 0
    a = (1.1 * np.log10(f) - 0.7)*h_R - (1.56*np.log10(f) - 0.8)
    L = []
    for d in distance:
        L.append(46.3 + 33.9 * np.log10(f) + 13.82 * np.log10(h_B) - a + (44.9 - 6.55 * np.log10(h_B)) * np.log10(d) + C)
    
    return L


def plot_network(dX, dY, X_bs, Y_bs, u_1, u_2):
    """ Plots """
#    fig = plt.gcf()
#    ax = fig.gca()
    plt.figure(figsize=(5,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlim(-dX / 2, dX / 2)
    plt.ylim(-dY / 2, dY / 2)

    plt.plot(u_1 - dX / 2, u_2 - dY / 2,'bo')    
    plt.plot(X_bs - dX / 2, Y_bs - dY / 2, 'ro')
    plt.grid(True)
    plt.title(r'\textbf{Base station and UE positions}')
    plt.xlabel('X pos (m)')
    plt.ylabel('Y pos (m)')
    plt.savefig('figures/network.pdf', format='pdf')
    plt.show()
    
    
def average_SINR_dB(random_state, g_ant=2, num_users=50, load=0.85, faulty_feeder_loss=0.0, beamforming=False, plot=False):

    np.random.seed(random_state)
    n = np.random.poisson(lam=num_users, size=None) # the Poisson random variable (i.e., the number of points inside ~ Poi(lambda))

    # a 5x5 room
    dX = 5.
    dY = 5.
    
    u_1 = np.random.uniform(0.0, dX, n) # generate n uniformly distributed points 
    u_2 = np.random.uniform(0.0, dY, n) # generate another n uniformly distributed points 
    
    # Now put a transmitter at point center
    X_bs = dX / 2.
    Y_bs = dY / 2.
    
    
    ######
    # Cartesian sampling
    #####
         
    
    # Distances in meters.
    dist = LA.norm((X_bs - u_1, Y_bs - u_2), axis=0)
    
    ptmax = 2 # in Watts
    
    # all in dB here
    #g_ant = 16
    ins_loss = 0.5
    g_ue = 0
    f = 28000 # 28 GHz
    h_R = 1.5 # human UE
    h_B = 3 # base station small cell
    NRB = 100 # Number of PRBs in the 20 MHz channel-- needs to be revalidated.
    B = 100e6 # 100 MHz
    T = 290 # Kelvins
    K = 1.38e-23
    path_loss = np.array(cost231(dist, f, h_R, h_B))
    #load = 1.0
    
    concurrent_users = 5 # 5 can be scheduled in any given time.. 1 is me and 4 are others.
    num_tx_antenna = 4 # actually this N_s = min(N_T, N_R)

    
    ptmax = 10*np.log10(ptmax * 1e3) # to dBm
    pt = ptmax - np.log10(NRB) - np.log10(12) + g_ant - ins_loss - faulty_feeder_loss
    pr_RE = pt - path_loss + g_ue # received reference element power (almost RSRP depending on # of antenna ports) in dBm.
    
    pr_RE_mW = 10 ** (pr_RE / 10) # in mW
        
    # Compute received SINR
    # Assumptions:
        # Signaling overhead is zero.  That is, all PRBs are data.
        # Interference is 100%
    SINR = []
    Nth = K * T * B * 1e3 # in mWatts

    equal_proportion = 1. / concurrent_users

    # MIMO or not... this is the question
    beamforming_gain = num_tx_antenna if (beamforming) else 1
    
    for i in np.arange(len(dist)):
        interference_i = 0
        for j in np.arange(len(dist)):
            interference_i += (pr_RE_mW[j] * load * NRB * equal_proportion * 12) if (abs(i - j) <= (concurrent_users - 1) // 2 and i != j) else 0 # assuming all UEs use 100% of NRBs
        SINR_i = beamforming_gain * pr_RE_mW[i] * load * NRB * equal_proportion * 12 / (Nth + interference_i)
        SINR.append(SINR_i)


   # SINR_dB = 10 * np.log10(SINR)
    
#    fig = plt.gcf()
#    p, x = np.histogram(SINR_dB)
#    p = np.insert(p, 0, 0)
#    plt.bar(x, p)
#    plt.grid()
    #plt.show()
    
    if (plot):
        plot_network(dX, dY, X_bs, Y_bs, u_1, u_2)
    
    SINR_average_dB = 10 * np.log10(np.mean(SINR))
    return SINR_average_dB

# The baseline measurement is computed:
baseline_SINR_dB = average_SINR_dB(plot=True, random_state=seed)

# Now start with few scenarios:

# - Increasing the antenna gain by 2 dB (done by user AI) - action 0
# - Randomly adding a new user (done by network)
# - Offload 30% of the users (done by user AI) - action 1
# - Feeder fault alarm (done by network)
# - Beamforming on (done by user AI) - action 2
# - Reducing the antenna gain by 2 dB (done by network)
# - Introducing a new site - action 3
# - Neighboring cell down (done by network)

state_count = action_count = 4

# Network
player_A_scenario_0_SINR_dB = average_SINR_dB(g_ant=16, num_users=65, load=0.85, faulty_feeder_loss=0.0, beamforming=False, random_state=seed) # reduces baseline.
player_A_scenario_1_SINR_dB = average_SINR_dB(g_ant=16, num_users=50, load=0.85, faulty_feeder_loss=3.0, beamforming=False, random_state=seed) # reduces baseline.
player_A_scenario_2_SINR_dB = average_SINR_dB(g_ant=14, num_users=50, load=0.85, faulty_feeder_loss=0.0, beamforming=False, random_state=seed) # reduces baseline.
player_A_scenario_3_SINR_dB = average_SINR_dB(g_ant=14, num_users=150, load=0.85, faulty_feeder_loss=0.0, beamforming=False, random_state=seed) # reduces baseline.

# RF engineer
player_B_scenario_0_SINR_dB = average_SINR_dB(g_ant=18, num_users=50, load=0.85, faulty_feeder_loss=0.0, beamforming=False, random_state=seed) # improves baseline.
player_B_scenario_1_SINR_dB = average_SINR_dB(g_ant=16, num_users=50, load=0.55, faulty_feeder_loss=0.0, beamforming=False, random_state=seed) # improves baseline.
player_B_scenario_2_SINR_dB = average_SINR_dB(g_ant=16, num_users=50, load=0.85, faulty_feeder_loss=0.0, beamforming=True, random_state=seed) # improves baseline.
player_B_scenario_3_SINR_dB = average_SINR_dB(g_ant=16, num_users=25, load=0.55, faulty_feeder_loss=0.0, beamforming=False, random_state=seed) # improves baseline.

player_A_rewards = [player_A_scenario_0_SINR_dB, player_A_scenario_1_SINR_dB, player_A_scenario_2_SINR_dB, player_A_scenario_3_SINR_dB]
player_B_rewards = [player_B_scenario_0_SINR_dB, player_B_scenario_1_SINR_dB, player_B_scenario_2_SINR_dB, player_B_scenario_3_SINR_dB]

# Time to build a reward martix 
# The rows are states and the columns are actions
R_A = np.empty([state_count, action_count])
R_B = np.empty([state_count, action_count])
for i in np.arange(state_count):
    for j in np.arange(state_count):
        R_A[j,i] = 1 if (i == j) else (player_A_rewards[i] - player_A_rewards[j])
        R_B[j,i] = -1 if (i == j) else (player_B_rewards[i] - player_B_rewards[j]) # the -1,+1 are actually infinity.

#print(R_A)
#print(R_B)

# Now proceed to Deep Q learning for Player B only
baseline_SINR_dB = 2
final_SINR_dB = baseline_SINR_dB + 5

env = SINR_environment(baseline_SINR_dB, final_SINR_dB, R_B, random_state=seed)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = DQNAgent(seed, state_size, action_size)

batch_size = 32
    
EPISODES = 100

plt.ion()
plt.figure()
plt.pause(0.001)

min_time = 501
best_episode = -1

for e in np.arange(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    score_progress = [baseline_SINR_dB]
    state_progress = ['start']
    action_progress = ['start']
    score = 0.
    for time in np.arange(500):
        # Let Network (Player A) function totally random
        state_space_network = np.arange(state_size)
        action_space_network = np.arange(action_size)
        state_network = np.random.choice(state_space_network)

        valid_move = (R_A[state_network] <= 0)
        if (any(valid_move)):
            np.random.shuffle(action_space_network)
            action_network = action_space_network[0]
            reward = R_A[state_network, action_network]
        else:
            reward = 0.


        score += reward # add the output of the network to the total score, which is negative
        action_progress.append('Network: Action {}'.format(action_network))
        
        score_progress.append(score) 
        
        # Now engineer (Player B) comes in.
        # env.render()
        action = agent.act(state)
        next_state, reward = env.step(action)
        done = (score >= final_SINR_dB)
        reward = reward if not done else 5. # game over and AI RF won.
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        score += reward # add the output of the user AI to the total score, which hopefully is positive.
        
        score_progress.append(score)
        
        state_progress.append(state[0])
        action_progress.append('Engineer AI: Action {}'.format(action))
            
        if done:
            if (time < min_time):
                best_episode = e
                min_time = time
            # Do some nice plotting here
            sinr_min = np.min(score_progress)
            sinr_max = np.max(score_progress)
            plt.figure()
            plt.plot(score_progress, marker='o', linestyle='--', color='b')
            plt.xlabel('Time / Epoch')
            plt.ylabel('Current SINR (dB)')
            plt.title('Episode {} / {}'.format(e + 1, EPISODES))
            plt.grid(True)
            plt.ylim((sinr_min - 1, sinr_max))
            if (e == 0):
                plt.savefig('figures/episode_0.pdf', format='pdf')
            if (e == EPISODES - 1):
                plt.savefig('figures/episode_final.pdf', format='pdf')
            if (e == best_episode):
                plt.savefig('figures/episode_best.pdf', format='pdf')
            plt.show()
            print("episode: {}/{}, score: {}, e: {:.2}"
              .format(e + 1, EPISODES, time, agent.epsilon))
            state_progress.append('end')
            action_progress.append('end')
            #print(state_progress)
            print('Action progress: ')
            print(action_progress)
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    if (e  + 1== 3):
        print(action_progress)
        print(score_progress)
        break
        
    if not done:
        print("episode: {}/{} failed to achieve target."
             .format(e + 1, EPISODES))
    
plt.ioff()
plt.show(block=True)



'''

######
# Polar sampling
#####
radii = np.zeros(n) # the radial coordinate of the points

for i in range(n):
    radii[i] = r * (np.sqrt(u_1[i]))

angle = np.zeros(n) # the angular coordinate of the points
for i in range(n):
    angle[i] = 2 * pi * u_2[i]

""" Plots """
fig = plt.gcf()
ax = fig.gca()
plt.xlim(-300, 300)
plt.ylim(-300, 300)
circ = plt.Circle((0, 0), radius=r, color='r', linewidth=2, fill=False)
# Cartesian Coordinates
x = np.zeros(n)
y = np.zeros(n)
for i in range(n):
    x[i] = radii[i] * np.cos(angle[i])
    y[i] = radii[i] * np.sin(angle[i])

plt.plot(x,y,'bo')
ax.add_artist(circ)
plt.show()
'''