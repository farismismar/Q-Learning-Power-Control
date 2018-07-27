#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:38:03 2018

@author: farismismar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import scipy.special
import matplotlib.ticker as tick

import os
os.chdir('/Users/farismismar/Desktop/E_Projects/UT Austin Ph.D. EE/Papers/4- Q-Learning Algorithm for VoLTE Closed-Loop Power Control in Indoor Small Cells')

qfunc = lambda x: 0.5-0.5*scipy.special.erf(x/np.sqrt(2))

# This has the Delta_gamma component
def ber_modified(sinr, delta=0, q=140):
    # sinr is in dB
    error = 1 - (1 - qfunc(np.sqrt(2.*(delta + 10**(sinr/10.))))) ** q # ** q
    return error


# sinr = np.linspace(0,18,100)
#per = [ber(x) for x in sinr]

#plt.figure(figsize=(7,5))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
##plot_edge, = plt.semilogy(sinr, ber_modified(sinr, delta=0, q=1), linestyle='-', color='k', label='QPSK (one OFDM symbol)')
#
#ax = plt.gca()
#ax.set_yscale('log')
#ax.get_xaxis().get_major_formatter().labelOnlyBase = False
#
#plot_baseline, = plt.semilogy(sinr, ber_modified(2.*sinr), linestyle='-', color='b', label='Average user $i$ (FPA)')
#
## Note the improvement was computed from Fig 11 in the paper.
##plot_vpc, = plt.semilogy(sinr, ber_modified(2.*sinr, 2/20+3*18/20), linestyle='-', color='r', label='Average user $i$ (Vanilla power control)')
#plot_dpc, = plt.semilogy(sinr, ber_modified(2.*sinr, 3*18/20), linestyle='-', color='g', label='Average user $i$ (DQN)')
#
#plt.grid(True,which="both")#,ls="-", color='0.65')
#
#plt.xlabel('Average DL SINR (dB)')
#plt.xlim(xmin=0,xmax=9)
#plt.ylabel('$\Xi$ PER')
#plt.title('Voice Packet Error Lower Bound Plot vs SINR -- One VoLTE Frame')
#
#plt.legend(handles=[plot_baseline, plot_dpc]) #plot_vpc, plot_dpc])
#plt.savefig('figures/packet_error.pdf', format="pdf")
#plt.show()
#plt.close()


####################################################################################
# 7/26/2018 run
####################################################################################
'''
Episode 879 finished after 16 timesteps (and epsilon = 0.01).
Network alarms progress: 
[0.0, -5.0, -0.0, -3.0, 0, 0.0, 0, 0, 0, -3.27, 0, 0, 0, 0, 0, 3.0]
PC state progress: 
['start', 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 'end']
PC action progress: 
[-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
SINR progress: 
[4.0, 3.0, -1.0, 0.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.73, 1.73, 2.73, 3.73, 4.73, 5.73, 8.73]
'''
'''
--------------------------------------------------------------------------------
Episode 879 finished after 20 timesteps.
Action progress: 
['start', 0, 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'network', 'end']
SINR progress: 
[4.0, 4.0, 4.0, 4.0, 4.0, 0.73, 0.73, 0.73, -2.27, 1.0, 1.0, -3, -3.0, -3.0, -3.0, -3.0, -3.0, 0.0, 0.0, 0.0, 0.0, 'end']
'''


# Plot PC for both algorithms
tau = 20
vanilla_tpc = [0,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0,0,0] # pad with zeros

time= np.arange(tau)

fig, ax1 = plt.subplots(figsize=(7,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.grid(True)
fpa = ax1.axhline(y=0, xmin=0, color="green", linewidth=1.5, label='Power commands -- FPA')
vanilla, = ax1.step(np.arange(len(vanilla_tpc)), vanilla_tpc, color='b', linewidth=2.5, label='Power commands -- Proposed')#
#deep, = ax1.step(np.arange(len(deepq_tpc)), deepq_tpc, color='b', label='Power commands -- DQN')#
ax1.set_xlabel('Transmit time interval (TTI)')
ax1.set_ylabel('Number of power commands')
ax1.set_yticks([-1,0,1,2,3])
ax1.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
ax1.xaxis.set_ticks(np.arange(0, tau + 1))
#ax2 = ax1.twinx()
#sinr, = plt.plot(time, SINR, linestyle='-', color='b', label='DL SINR')
plt.title(r'Power Commands ')
plt.legend(handles=[fpa, vanilla])# vanilla, deep])
#ax2.set_ylabel('Average DL SINR $\gamma_{DL}$(dB)')

plt.xlim(xmin=0,xmax=tau)

fig.tight_layout()
plt.savefig('figures/tpc.pdf', format="pdf")
plt.show(fig)
plt.close(fig)

####################################################################################
# Plot MOS
tau = 20 # ms
T = tau #6 * 1e3 * tau # 120 sec

sinr = np.linspace(-2,14,100)

#def scale(per):  
    #per_scaled = [0.005 if i < -2 else (0.5 * (-2 - i) / (-2 - max(per))) for i in per]
#    per_scaled = [0.005 + i for i in per_scaled] # 0.05% is the minimum value
    
 #   return per_scaled

# Obtain the corrective/improvement factors from the PC plot before you run this snipped
def payload(T, tau=20, NAF=0.5, Lamr=0, Lsid=61): # T and tau in ms, Lamr/Lsid is in bits
    Lsid = Lsid * tau / 8  # from bits to bytes per sec
    return NAF * Lamr * np.ceil(T/tau) + (1 - NAF) * Lsid * np.ceil(T/(8*tau))

fig = plt.figure(figsize=(7,5))
for improvement in np.array([0, -1 * 1/20 +1 * 15/20 ]): # 0 improvement is the : 
    result = []
    
    for framelength in np.arange(1000): # 1000 taus
        volterate = 23.85 # kbps
        NAF = 0.7
        
        ber = [ber_modified(x, delta=improvement, q=7*framelength*tau) for x in sinr]  # for bit error rate, we compute number of REs per frame hence * 7
        per = np.round(np.log10(ber), 0) # we actually need the exponent.
        N = len(per)
        fer = sum([1 for x in per if x > -2]) / N #scale(per)  # will get a frame error for this bit error rate
        Lamr = volterate * tau / 8 # in bytes per sec
        payld = payload(T=tau, tau=tau, NAF=NAF, Lamr=Lamr) # in bytes
        MOS = 4 - 0.7*np.log(fer) - 0.1 * np.log(tau * framelength * fer * N) # fer ok, second term: duration of lost packets in ms?

        MOS = min(4, MOS) #4 if MOS >= 4 else MOS # for i in MOS]
        MOS = max(1, MOS) #1 if MOS <= 1 else MOS# for i in MOS]
    #    print(improvement, fer, MOS)
        result.append(MOS)
    
        #print('{} {:.0f}% {:.1f}'.format(volterate, 100 * NAF, payld))
        if (improvement == 0):
            str = 'FPA'
        else: 
            str = 'Proposed'
    plt.plot(result, label='AMR = {} kbps, AF = {}, Power control = {}'.format(volterate, NAF, str))
    
    
plt.legend()
plt.title('Experimental mean opinion score vs packet error rate')
plt.xlabel('Packet error rate')
plt.ylabel('MOS')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlim(xmin=0,xmax=250)

plt.grid(True)

# Fix the x axis to show packet error rates 
ax = plt.gca()
#ax.set_xticks([0,200,400,600,800,1000])
ax.set_xticks([0,50,100,150,200,250])
#ax.set_xticklabels([0,0.1,0.2,0.3,0.4,0.5])
ax.set_xticklabels([0,0.05,0.10,0.15,0.2,0.25])


plt.savefig('figures/mos.pdf', format='pdf')
plt.show()
plt.close(fig)



####################################
# Plotting the episodes on one graph
####################################
SINR_MIN = -3 #dB  
baseline_SINR_dB = 4.0
final_SINR_dB = baseline_SINR_dB + 2.0 # this is the improvement
max_timesteps_per_episode = 20

score_progress_cl = [4.0, 3.0, -1.0, 0.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.73, 1.73, 2.73, 3.73, 4.73, 5.73, 8.73]
score_progress_fpa = [4.0, 4.0, 4.0, 4.0, 4.0, 0.73, 0.73, 0.73, -2.27, 1.0, 1.0, -3, -3.0, -3.0, -3.0, -3.0, -3.0, 0.0, 0.0, 0.0, 0.0]


# Do some nice plotting here
fig = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel('Transmit Time Intervals (1 ms)')

# Only integers                                
ax = fig.gca()
ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode + 1))

ax.set_autoscaley_on(False)

plt.plot(score_progress_fpa, marker='o', linestyle=':', color='b', label='FPA')
plt.plot(score_progress_cl, marker='D', linestyle='-', color='k', label='Proposed')

plt.xlim(xmin=0, xmax=max_timesteps_per_episode)

plt.axhline(y=SINR_MIN, xmin=0, color="red", linewidth=1.5)
plt.axhline(y=final_SINR_dB, xmin=0, color="green",  linewidth=1.5)
plt.ylabel('Average DL Received SINR (dB)')
plt.title('Final Episode')
plt.grid(True)
plt.ylim(-8,10)
plt.legend()

plt.savefig('figures/episode_final_output.pdf', format="pdf")
plt.show(block=True)
plt.close(fig)

