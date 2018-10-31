#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:20:05 2017

@author: farismismar
"""

import random
import numpy as np
#from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as tick

from environment import SINR_environment
from QLearningAgent import QLearningAgent as QLearner
#from DQNLearningAgent import DQNLearningAgent as QLearner

SINR_MIN = -3 #dB  
baseline_SINR_dB = 4.0
final_SINR_dB = baseline_SINR_dB + 2.0 # this is the improvement
MAX_EPISODES = 707 # successful ones [85, 115, 129, 258, 259, 284, 285, 286, 707]
N_interferers = 4 # this is a hardcoded parameter: 4 base stations.

L_geom = 10. #  meters

pt_max = 2 # in Watts

# all in dB here
g_ant = 2
ins_loss = 0.5
g_ue = 0
f = 2600 
h_R = 1.5 # human UE
h_B = 3 # base station small cell height
NRB = 100 # Number of PRBs in the 20 MHz channel-- needs to be revalidated.
B = 20e6 # 20 MHz
T = 290 # Kelvins/
K = 1.38e-23
num_antenna_streams = None # Not needed
N0 = K*T*B*1e3 # Thermal noise.  Assume Noise Figure = 0 dB.


import os
os.chdir('/Users/farismismar/Desktop/4- Q-Learning Algorithm for VoLTE Closed-Loop Power Control in Indoor Small Cells')

seed = 0 

random.seed(seed)
np.random.seed(seed)

def cost231(distance, f, h_R, h_B):
    C = 0
    a = (1.1 * np.log10(f) - 0.7)*h_R - (1.56*np.log10(f) - 0.8)
    L = []
    for d in distance:
        L.append(46.3 + 33.9 * np.log10(f) + 13.82 * np.log10(h_B) - a + (44.9 - 6.55 * np.log10(h_B)) * np.log10(d) + C)
    
    return L # in dB


def plot_pc_actions(tpc, episode):
    tau = 20

    tpc.insert(0, 0)
    tpc = np.array(tpc)
   
    fig, ax1 = plt.subplots(figsize=(7,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.grid(True)
    fpa = ax1.axhline(y=0, xmin=0, color="green", linewidth=1.5, label='Power commands -- FPA')
    closedloop, = ax1.step(np.arange(len(tpc)), tpc, color='b', label='Power commands -- Proposed')#
    ax1.set_xlabel('Transmit Time Interval (1 ms)')
    ax1.set_ylabel('Power commands (dB)')
    ax1.set_yticks([-3,-2,-1,0,1,2,3])
    ax1.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
    ax1.xaxis.set_ticks(np.arange(0, tau + 1))
    plt.title(r'Episode $\tau = {}$ -- Power Commands '.format(episode))
    plt.legend(handles=[fpa, closedloop])# vanilla, deep])
    
    plt.xlim(xmin=0,xmax=tau)
    
    fig.tight_layout()
    plt.savefig('figures/tpc_{}.pdf'.format(episode), format="pdf")
    plt.show(fig)
    plt.close(fig)


def plot_network(dX, dY, X_bs, Y_bs, u_1, u_2):
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

def compute_interference_power(base_station_id, UE_x, UE_y):
    # Returns the received interference power in mW as measured by the UE at location (x,y)
    # The four interfering base stations.
    if (base_station_id == 0):
        X_bs = -L_geom
        Y_bs = 0
    if (base_station_id == 1):
        X_bs = 0
        Y_bs = L_geom
    if (base_station_id == 2):
        X_bs = 0
        Y_bs = -L_geom
    if (base_station_id == 3):
        X_bs = L_geom
        Y_bs = 0
        
    # Distances in kilometers.
    UE_x = np.array(UE_x)
    UE_y = np.array(UE_y)
   
    dist = np.power((np.power(X_bs-UE_x, 2) + np.power(Y_bs-UE_y, 2)), 0.5) / 1000.
    recv_power = pt_max * 1e3 / np.array([10 ** (l / 10.) for l in cost231(dist, f, h_R, h_B)]) # in mW
    
    average_recv_power = sum(recv_power) / len(recv_power)
    return average_recv_power

def average_SINR_dB(random_state, g_ant=2, num_users=50, load=0.7, faulty_feeder_loss=0.0, beamforming=False, plot=False):

    np.random.seed(random_state)
    n = np.random.poisson(lam=num_users, size=None) # the Poisson random variable (i.e., the number of points inside ~ Poi(lambda))

    dX = L_geom
    dY = L_geom
    
    u_1 = np.random.uniform(0.0, dX, n) # generate n uniformly distributed points 
    u_2 = np.random.uniform(0.0, dY, n) # generate another n uniformly distributed points 
    
    # Now put a transmitter at point center
    X_bs = dX / 2.
    Y_bs = dY / 2.
        
    ######
    # Cartesian sampling
    #####
         
    # Distances in kilometers.
    dist = np.power((np.power(X_bs-u_1, 2) + np.power(Y_bs-u_2, 2)), 0.5) / 1000. #LA.norm((X_bs - u_1, Y_bs - u_2), axis=0) / 1000.

    path_loss = np.array(cost231(dist, f, h_R, h_B))
    
    ptmax = 10*np.log10(pt_max * 1e3) # to dBm
    pt = ptmax - np.log10(NRB) + g_ant - ins_loss - faulty_feeder_loss #+ 10 * np.log10(num_antenna_streams) # - np.log10(12)
    pr_RB = pt - path_loss + g_ue # received reference element power (almost RSRP depending on # of antenna ports) in dBm.
  
    pr_RB_mW = 10 ** (pr_RB / 10) # in mW the total power of the PRB used to transmit packets
    
    # Compute received SINR
    # Assumptions:
    # Signaling overhead is zero.  That is, all PRBs are data.
    
    SINR = []

    # Generate some thermal noise
    SINR = []
    for i in np.arange(len(dist)): # dist is a distance vector of UE i
        ICI = 0 # interference on the ith user
        for j in np.arange(N_interferers):
            ICI += compute_interference_power(j, u_1, u_2)

        SINR_i = pr_RB_mW[i] / (N0 + ICI)
        SINR.append(SINR_i)
    
    if (plot):
        plot_network(dX, dY, X_bs, Y_bs, u_1, u_2)
    
    SINR_average_dB = 10 * np.log10(np.mean(SINR))
    return SINR_average_dB

# This formula is from Computation of contribution of action νin= 2 in paper.
def compute_SINR_due_to_neighbor_loss():
    numerator =  10**((pt_max - np.log10(NRB) + g_ant - ins_loss)/10)
    denom = (N0 + (N_interferers - 1) * pt_max)
    return 10*np.log10(numerator / denom)

################ Actions for Power Control ###################
# 0- Cluster is normal
# 1- Feeder fault alarm (3 dB)
# 2- Neighboring cell down
# 3- VSWR out of range alarm: https://antennatestlab.com/antenna-education-tutorials/return-loss-vswr-explained
# 4- Clear 1 ( - dB losses)
# 5- Clear 2
# 6- Clear 3
    
 
state_count = 3 # ok we agree on this
action_count_a = 11 # this is the upper index of player A index.
action_count_b = 5 # this is per player_B_contribs.

# Network
player_A_scenario_0_SINR_dB = average_SINR_dB(g_ant=16, num_users=10, faulty_feeder_loss=0.0, beamforming=False, random_state=seed)
player_A_scenario_1_SINR_dB = average_SINR_dB(g_ant=16, num_users=10, faulty_feeder_loss=3.0, beamforming=False, random_state=seed)
player_A_scenario_2_SINR_dB = compute_SINR_due_to_neighbor_loss();
player_A_scenario_3_SINR_dB = average_SINR_dB(g_ant=16, num_users=10, faulty_feeder_loss=5, beamforming=False, random_state=seed) #  VSWR of 1.4:1 too bad.  We want VSWR 1.2:1.  This is 5 dB

player_A_SINRs = [player_A_scenario_0_SINR_dB, player_A_scenario_1_SINR_dB, player_A_scenario_2_SINR_dB, player_A_scenario_3_SINR_dB]

player_A_contribs = player_A_SINRs - player_A_scenario_0_SINR_dB
player_A_contribs[2] = player_A_scenario_2_SINR_dB
player_A_contribs = np.append(-player_A_contribs, player_A_contribs)
player_B_contribs = np.array([0, -3, -1, 1, 3]) # TPCs (up to 2 TPCs per TTI)
player_A_contribs = np.append([0,0,0], player_A_contribs)
alarm_reg = [0,0,0]

batch_size = 32 
R_max = 2.
R_min = -100.
env = SINR_environment(baseline_SINR_dB, final_SINR_dB, seed)
agent = QLearner(seed=seed, state_size=state_count, action_size=action_count_b, batch_size=batch_size)
succ = [] # a list to save the good episodes


def get_A_contrib():
    # Draw a number at random
    n = np.random.randint(action_count_a)

    # if n is less than 4, then it is a Normal action, let it go through..  
    if n < 4:
        return player_A_contribs[n]
    
    if n < 7: # this is a clear alarm  (8,9,10)--- only clear it if the alarm has been set, otherwise, return no change
        if (alarm_reg[n - 4] == 1):
            alarm_reg[n - 4] = 0 # alarm has been cleared.
            return player_A_contribs[n]       
        else:
            return 0
    else: 
        # an alarm
        if (alarm_reg[n - 8] == 0):
            alarm_reg[n - 8] = 1 #  set up alarm in register.            
            return player_A_contribs[n]
        else:
            return 0

    return None

def run_agent(env, plotting = False):
    global alarm_reg
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = 20 # one AMR frame ms.

    retainability = [] # overall
  
    successful = False
    for episode_index in np.arange(max_episodes_to_run):
        state = env.reset()
        reward = R_min
        action = agent.begin_episode(state)
    
        cell_score = baseline_SINR_dB
        pt_current = 0.1 # in Watts, initial transmit power.

        # Recording arrays
        state_progress = ['start', 0]
        action_progress = ['start', 0]
        score_progress = [cell_score]
        alarm_reg = [0,0,0]
        pc_progress = []        
        network_progress = []
        
        for timestep_index in range(max_timesteps_per_episode):
            # Let Network (Player A) function totally random
            network_issue = get_A_contrib()
            cell_score += network_issue #player_A_contribs[np.random.randint(action_count_a)]
            action_progress.append('network')  # The network action is empty
            
            # Perform the power control action and observe the new state.
            action = agent.act(state, reward)
            next_state, reward, _, _ = env.step(action)
            power_command = player_B_contribs[action]

            pt_current *= 10 ** (power_command / 10.) # the current ptransmit in mW due to PC
            
            # ptransmit cannot exceed pt, max.
            if (pt_current >= pt_max):
                pt_current = pt_max
            else:
                cell_score += power_command            

            aborted = (cell_score < SINR_MIN)
            done = (cell_score >= final_SINR_dB)
            
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state
        
            # Rewards are here
            if done:
                if timestep_index < 15: # premature ending -- cannot finish sooner than 15 episodes
                    aborted = True
                    successful = False
                else:                       # ending within time.
                   successful = True
                   reward = R_max
                   aborted = False
            
            #print(reward)
#            
            # I truly care about the net change: network - PC
            action_progress.append(action)
            network_progress.append(np.round(network_issue,2))
            pc_progress.append(np.round(power_command,2))
            score_progress.append(np.round(cell_score, 2))
            state_progress.append(np.round(next_state[0], 2))            
            retainability.append(np.round(cell_score, 2))
                                
            if aborted == True:
                reward = R_min
                state_progress.append('ABORTED')
                
            if (done or aborted):
                print("Episode {0} finished after {1} timesteps (and epsilon = {2:0.3}).".format(episode_index + 1, timestep_index + 1, agent.exploration_rate))
                state_progress.append('end')
                print('Network alarms progress: ')
                print(network_progress)    
                print('PC state progress: ')
                print(state_progress)                
                print('PC action progress: ')
                print(pc_progress)
                print('SINR progress: ')
                print(score_progress) # this is actually the SINR progress due to the score or after both player A and B have played.
                
                if (plotting):
                     # Do some nice plotting here
                    fig = plt.figure()
    
                    plt.rc('text', usetex=True)
                    plt.rc('font', family='serif')
                    plt.xlabel('Transmit Time Interval (1 ms)')
    
                    # Only integers                                
                    ax = fig.gca()
                    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
                    ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode + 1))
    
                    ax.set_autoscaley_on(False)
    
                    plt.plot(score_progress, marker='o', linestyle='--', color='b')
                    plt.xlim(xmin=0, xmax=max_timesteps_per_episode)
    
                    plt.axhline(y=SINR_MIN, xmin=0, color="red", linewidth=1.5)
                    plt.axhline(y=final_SINR_dB, xmin=0, color="green",  linewidth=1.5)
                    plt.ylabel('Average DL Received SINR (dB)')
                    plt.title('Episode {0} / {1} ($\epsilon = {2:0.3}$)'.format(episode_index + 1, max_episodes_to_run, agent.exploration_rate))
                    plt.grid(True)
                    plt.ylim(-8,10)
                    plt.savefig('figures/episode_{}.pdf'.format(episode_index + 1), format="pdf")
                    if (plotting):
                        plt.show(block=True)
                    plt.close(fig)
                    
                    plot_pc_actions(pc_progress, episode_index + 1)
                    
                print('-'*80)       
                break                    

        if (successful):
            succ.append(episode_index+1)
        
        # For multi-plotting purposes
        if (episode_index + 1 == 725 or episode_index + 1 == 2): # 260 398
            file = open("plot_sinr.txt","a") 
            for item in score_progress:
                file.write("{},".format(item))
            file.write("\n")
            file.close()

#        # Remove these four lines after finding the correct episode            
#        if (successful and episode_index + 1 >= 2000):
#   #         break # This is the number I truly need to run my program for.
#        #We found xxx episodes required.
#            None
#        else:
#            successful = False

        # train the agent with the experience of the episode
        if len(agent.memory) > agent.batch_size:
            agent.replay(agent.batch_size)
            # Show the losses here
            losses = agent.get_losses()
            
    print('Overall retainability')
    dcr =sum(1 for i in retainability if i < 0) / len(retainability)
    retainability = 1. - dcr
    print('{}%'.format(100*retainability))
            
    # Now plot the losses NOT for vanilla Q
    try:
        print('Successful episodes')
        print(succ)
        
        if len(losses) > 0:
            fig = plt.figure(figsize=(7,5))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.plot(losses, color='k')
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.title('Losses vs Episode')
            
            ax = fig.gca()
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
        
            plt.grid(True)
            plt.ylim(ymin=-400)
            plt.axhline(y=0, xmin=0, color="gray", linestyle='dashed', linewidth=1.5)
            plt.savefig('figures/loss_episode.pdf', format="pdf")
            if (plotting):
                plt.show(block=True)
            plt.close(fig)
    except:
        None # technically do nothing-- this is the vanilla Q
        
    if (not successful):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum.".format(max_episodes_to_run))
    
    #plt.ioff()


def run_agent_fpa(env, plotting = False):
    global alarm_reg 
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = 20 # one AMR frame.

    for episode_index in np.arange(max_episodes_to_run):    
        cell_score = baseline_SINR_dB

        # Recording arrays
        #state_progress = ['start', 0]
       # action_progress = ['start', 0]
        score_progress = [cell_score]
        alarm_reg = [0,0,0]      
      #  network_progress = []
        retainability = [] # overall
        for timestep_index in range(max_timesteps_per_episode):
            # Let Network (Player A) function totally random
            network_issue = get_A_contrib()
            cell_score += network_issue #player_A_contrib
           # action_progress.append('network')  # The network action is empty
            
            if (cell_score < SINR_MIN):
                cell_score = SINR_MIN
            
            done = (timestep_index == max_timesteps_per_episode - 1)

            # I truly care about the net change: network - PC
           # action_progress.append(np.round(cell_score, 2))
           # network_progress.append(np.round(network_issue,2))
#            pc_progress.append(np.round(power_command,2))
            score_progress.append(np.round(cell_score, 2))
 #           state_progress.append(np.round(next_state[0], 2))            
            retainability.append(np.round(cell_score, 2))
                                                
            if (done): # or aborted):
                print("Episode {0} finished after {1} timesteps.".format(episode_index + 1, timestep_index + 1))
                #finished = True
                score_progress.append('end')
                
                print('SINR progress: ')
                print(score_progress)    
               
                # Do some nice plotting here
                if (plotting):
                    fig = plt.figure()
                    plt.rc('text', usetex=True)
                    plt.rc('font', family='serif')
                    plt.xlabel('Transmit Time Intervals (1 ms)')
    
                    # Only integers                                
                    ax = fig.gca()
                    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
                    ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode + 1))
    
                    ax.set_autoscaley_on(False)
    
                    plt.plot(score_progress, marker='o', linestyle='--', color='b')
                    plt.xlim(xmin=0, xmax=max_timesteps_per_episode)
    
                    plt.axhline(y=SINR_MIN, xmin=0, color="red", linewidth=1.5)
                    plt.axhline(y=final_SINR_dB, xmin=0, color="green",  linewidth=1.5)
                    plt.ylabel('Average DL Received SINR (dB)')
                    plt.title('Episode {0} / {1}'.format(episode_index + 1, max_episodes_to_run))
                    plt.grid(True)
                    plt.ylim(-8,10)
                    
                    plt.savefig('figures_fpa/episode_{}.pdf'.format(episode_index + 1), format="pdf")
                    plt.show(block=True)
                    plt.close(fig)
                    
                print('-'*80)       
                break                    
#        # For multi-plotting purposes
#        if (episode_index + 1 == 879):# or episode_index + 1 == 2):
#            file = open("plot_sinr_fpa.txt","a") 
#            for item in score_progress:
#                file.write("{},".format(item))
#            file.write("\n")
#            file.close()

            
    print('Overall retainability')
    dcr =sum(1 for i in retainability if i < 0) / len(retainability)
    retainability = 1. - dcr
    print('{}%'.format(100*retainability))
    
########################################################################################
    
#run_agent(env, True)  # Overall retainability 78.75%  <- to obtain, run again and fix the max episode to the optimal
run_agent_fpa(env, False) # 55.00%

########################################################################################
