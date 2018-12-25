#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:38:03 2018

@author: farismismar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import scipy.special
import matplotlib.ticker as tick

import os
os.chdir('/Users/farismismar/Desktop/E_Projects/UT Austin Ph.D. EE/Papers/Journals/2- A Framework for Automated Cellular Network Tuning with Reinforcement Learning/Major Revision 12-2018 Submission/Simulation')

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
##plot_vpc, = plt.semilogy(sinr, ber_modified(2.*sinr, 2/20+3*18/20), linestyle='-', color='r', label='Average user $i$ (proposed power control)')
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

#######
# 10/30/2018 run for Asilomar
'''
Episode 707 finished after 16 timesteps (and epsilon = 0.01).
Network alarms progress: 
[0, 0.0, 0.0, -3.0, 0, 0.0, -0.0, 0.0, 0, 0, -6.28, 0, -0.0, 3.0, 0, 6.28]
PC state progress: 
['start', 0, 2, 1, 0, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 'end']
PC action progress: 
[-1, 1, 0, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
SINR progress: 
[4.0, 3.0, 4.0, 4.0, -0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.28, -0.28, 0.72, 4.72, 5.72, 13.0]
[4.0, 3.0, 4.0, 4.0, -0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.28, -0.28, 0.72, 4.72, 5.72, 6.0]  <- capping the max to the upper bound for journal submission.

FPA: 
    Episode 707 finished after 20 timesteps.
SINR progress: 
[4.0, 4.0, 1.0, 1.0, 1.0, 1.0, -3, -3, -3, -3.0, -3.0, -3.0, -3.0, 2.0, -3.0, 2.0, 2.0, -3, 0.0, 0.0, 0.0, 'end']

'''

####################################################################################
# Plot MOS
tau = 20 # ms
T = tau #6 * 1e3 * tau # 120 sec

sinr = np.linspace(-2,14,100)


# Obtain the corrective/improvement factors from the PC plot before you run this snipped
def payload(T, tau=20, NAF=0.5, Lamr=0, Lsid=61): # T and tau in ms, Lamr/Lsid is in bits
    Lsid = Lsid * tau / 8  # from bits to bytes per sec
    return NAF * Lamr * np.ceil(T/tau) + (1 - NAF) * Lsid * np.ceil(T/(8*tau))

fig = plt.figure(figsize=(8,5))
params = {'backend': 'ps',
          'axes.labelsize': 12, # fontsize for x and y labels (was 10)
          'axes.titlesize': 12,
          'font.size': 10, # was 10
          'legend.fontsize': 12, # was 10
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'font.family': 'serif'
}

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rc('figure', titlesize=10) 
matplotlib.rcParams.update(params)

# TODO: compute improvement due to PC as a weighted av
for improvement in np.array([0, 12/20*1 + 3/20*2 + 5/20*0, 2]): # find this ratio as a weighted av from PC 
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
        style = '-'
        if (improvement == 0):
            str = 'FPA'
        elif (improvement == 2): 
            str = 'Maximum SINR'
            style = '--'
        else:
            str = 'Proposed'
    #plt.plot(result, label='AMR = {} kbps, AF = {}, Power control = {}'.format(volterate, NAF, str))
    plt.plot(result, linestyle=style, label='Power control = {}'.format(str))
    
    
plt.legend()
plt.title(r'\textbf{Experimental MOS}')
plt.xlabel(r'Packet error rate')
plt.ylabel(r'MOS')
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
xi = 2.0 # this is the improvement
final_SINR_dB = baseline_SINR_dB + xi
max_timesteps_per_episode = 20

# TODO: Fill in vectors
score_progress_cl = [4.0, 3.0, 4.0, 4.0, -0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.28, -0.28, 0.72, 4.72, 5.72, 6.0]
score_progress_fpa = [4.0, 4.0, 1.0, 1.0, 1.0, 1.0, -3, -3, -3, -3.0, -3.0, -3.0, -3.0, 2.0, -3.0, 2.0, 2.0, -3, 0.0, 0.0, 0.0]

# This vector does not need to be filled in.
score_progress_max = np.array(baseline_SINR_dB)
score_progress_max = np.insert(score_progress_max, 1, final_SINR_dB * np.ones(len(score_progress_fpa) - 1))

# Do some nice plotting here
fig = plt.figure(figsize=(8,5))

params = {'backend': 'ps',
          'axes.labelsize': 10, # fontsize for x and y labels (was 10)
          'axes.titlesize': 12,
          'font.size': 10, # was 10
          'legend.fontsize': 12, # was 10
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'font.family': 'serif'
}

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rc('figure', titlesize=10) 
matplotlib.rcParams.update(params)

plt.xlabel(r'$\tau$ (1 ms)')

# Only integers
ax = fig.gca()
ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode + 1))

ax.set_autoscaley_on(False)

plt.plot(score_progress_fpa, marker='o', linestyle=':', color='b', label='FPA')
plt.plot(score_progress_cl, marker='D', linestyle='-', color='k', label='Proposed')
plt.plot(score_progress_max, marker='+', linestyle='--', color='m', label='Maximum SINR')

plt.xlim(xmin=0, xmax=max_timesteps_per_episode)

plt.axhline(y=final_SINR_dB, xmin=0, color="green", linewidth=1.5)
plt.axhline(y=SINR_MIN, xmin=0, color="red", linewidth=1.5)
plt.ylabel(r'Average DL Received SINR (dB)')
plt.ylabel(r'$\bar\gamma_\text{DL}[t]$ (dB)')
plt.title(r'\textbf{\Large Final episode}')
plt.grid(True)
plt.ylim(-8,10)
plt.legend()

plt.savefig('figures/episode_final_output.pdf', format="pdf")
plt.show(block=True)
plt.close(fig)


####################################################################################
# Plot PC for all algorithms
# TODO: Obtain TPC from the proposed PC
tau = 20
tpc_cl = [0, 2, 1, 0, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,0,0,0] # pad with zeros

time= np.arange(tau)

# Do some nice plotting here
fig, ax1 = plt.subplots(figsize=(8,5))

params = {'backend': 'ps',
          'axes.labelsize': 10, # fontsize for x and y labels (was 10)
          'axes.titlesize': 12,
          'font.size': 10, # was 10
          'legend.fontsize': 12, # was 10
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'font.family': 'serif'
}

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rc('figure', titlesize=10) 
matplotlib.rcParams.update(params)


#tpc_upper = np.round(tpc_upper,0) 

plt.grid(True)
fpa = ax1.axhline(y=0, xmin=0, color="red", linewidth=1.5, label='Power commands -- FPA')
proposed, = ax1.step(np.arange(len(tpc_cl)), tpc_cl, color='b', linewidth=2.5, label='Power commands -- Proposed')#
upper = ax1.axhline(y=xi, xmin=0, color="green", linestyle='--', linewidth=1.5, label='Power commands -- Maximum SINR (upper bound)')


ax1.set_xlabel(r'$\tau$ (1 ms)')
ax1.set_ylabel(r'$\kappa[t]\text{PC}[t]$ (dB)')
ax1.set_yticks(np.linspace(-2,3,num=6))
ax1.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
ax1.xaxis.set_ticks(np.arange(0, tau + 1))
plt.title(r'\textbf{\Large Power commands}')
plt.legend(handles=[fpa, proposed, upper])

plt.xlim(xmin=0,xmax=tau)

fig.tight_layout()
plt.savefig('figures/tpc.pdf', format="pdf")
plt.show(fig)
plt.close(fig)

