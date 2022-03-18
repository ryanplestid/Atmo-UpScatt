'''
Example showing how to calculate the bounds
    for a desired number of visible HNL events.
This method calculates a lower bound and a right
    bound and pieces them together.  This works well
    when the bounds are shaped similarly to our 
    dipole bounds.
'''

import numpy as np
from numpy import sin, cos, pi
from numpy import random as rand 
import scipy as sp
from scipy import special as spc
import matplotlib as mpl #Package to help with plotting
from matplotlib import pyplot as plt
plt.style.use(['science','ieee'])

#Import modules that we made
import MainIntegration
import reweightModule

#Run the Monte Carlo for d and m_N values around the
#   area where we are looking for bounds
MC_m_N_vals = np.logspace(-2.5,-.1,12)
MC_d_vals = np.logspace(-11,-8,12)
num_events = int(2e4)

filename,Decays = MainIntegration.MonteCarlo(MC_m_N_vals, MC_d_vals, num_events)

#Specify the properties for our experiment
V_det = 22500 * 1e6 #Detector volume (cm^3)
alpha_decay = 0 #0 for Majoranna HNL, [-1,1] for Dirac
flavors = ['E', 'EBar','Mu', 'MuBar', 'Tau', 'TauBar'] 
#neutrino flavors coupled to HNL
total_time = 10 * 365 * 86400 #total live time (in s)
desired_excess = 100

#Find the lower bound
#   specify m_N values, and find corresponding d values
lower_m_N_vals = np.logspace(-2,-1,6)
lower_d_vals = np.zeros(6)
index = 0

starting_d_limits = [10**(-10.5), 10**(-8.5)] #minimum and maximum values of d
outside_bound_indices = np.array([])

for m_N in lower_m_N_vals:
    print('m_N', m_N)
    converged = False
    d_limits = starting_d_limits
    
    while converged == False:
        d = np.sqrt(d_limits[0]*d_limits[1])
        print('d',d)
        #Find Monte Carlo simulation that had closest parameters
        least_greater_m_N_index = int(np.where((MC_m_N_vals - m_N) > 0)[0][0])
        least_greater_m_N = MC_m_N_vals[least_greater_m_N_index]
        try:
            nearest_d_index = int(np.where(abs(MC_d_vals - d) == min(abs(MC_d_vals - d)))[0])
        except:
            nearest_d_index = int(np.where(abs(MC_d_vals -d) > 0)[0][0])
        nearest_d = MC_d_vals[nearest_d_index]
        
        filename = "mN_%.3g" %least_greater_m_N+"_d_%.3g"%nearest_d+"exp_other.events"
        #If you change the file naming formalism, you'll have to change this too
        threshold = 0.05
        reweightModule.Update_Fluxes(filename,threshold)
        
        Rate,rate_error,cos_phi_midpoints, angular_rates, \
            E_gamma_midpoints,E_gamma_rates,high_E_rate \
                =reweightModule.ReIntegrate(filename,nearest_d,least_greater_m_N,\
                                            alpha_decay,V_det,flavors)
        
        #Check to see if we've reached our desired value
        if abs(Rate * total_time - desired_excess) < 1:
            lower_d_vals[index] = d
            converged = True
        elif Rate*total_time > desired_excess:
            d_limits[1] = d
        elif Rate*total_time < desired_excess:
            d_limits[0] = d
        if (d_limits[1]/d_limits[0]) < 1.01:
            lower_d_vals[index] = d
            converged = True
            
            #Check if it converged on a max of min
            if d_limits[0] == starting_d_limits[0] or d_limits[1]== starting_d_limits[1]:
                outside_bound_indeces = np.append(outside_bound_indices,index)
        
    index += 1
#Cut out values that did not converge
try:
    cut = min(outside_bound_indices)
except:
    cut = len(lower_m_N_vals)

lower_m_N_vals = lower_m_N_vals[0:cut]
lower_d_vals = lower_d_vals[0:cut]

#Calculate Right/Upper Bounds
#   specify d values, and find corresponding m_N values
upper_m_N_vals = np.zeros(6)
upper_d_vals = np.logspace(-9.5,-8.5,6)
index = 0

starting_m_N_limits = [0.03, 0.3] #minimum and maximum values of m_N
outside_bound_indices = np.array([])

for d in upper_d_vals:
    print('d', d)
    converged = False
    m_N_limits = starting_m_N_limits
    
    while converged == False:
        m_N = np.sqrt(m_N_limits[0]*m_N_limits[1])
        print('m_N', m_N)
        #Find Monte Carlo simulation that had closest parameters
        least_greater_m_N_index = int(np.where((MC_m_N_vals - m_N) > 0)[0][0])
        least_greater_m_N = MC_m_N_vals[least_greater_m_N_index]
        nearest_d_index = int(np.where(abs(MC_d_vals - d) == min(abs(MC_d_vals - d)))[0])
        nearest_d = MC_d_vals[nearest_d_index]
        
        filename = "mN_%.3g" %least_greater_m_N+"_d_%.3g"%nearest_d+"exp_other.events"
        #If you change the file naming formalism, you'll have to change this too
        threshold = 0.05
        reweightModule.Update_Fluxes(filename,threshold)
        
        Rate,rate_error,cos_phi_midpoints, angular_rates, \
            E_gamma_midpoints,E_gamma_rates,high_E_rate \
                =reweightModule.ReIntegrate(filename,nearest_d,least_greater_m_N,\
                                            alpha_decay,V_det,flavors)
        
        #Check to see if we've reached our desired value
        if abs(Rate * total_time - desired_excess) < 1:
            upper_m_N_vals[index] = m_N
            converged = True
        elif Rate*total_time > desired_excess:
            m_N_limits[0] = m_N
        elif Rate*total_time < desired_excess:
            m_N_limits[1] = m_N
        if (m_N_limits[1]/m_N_limits[0]) < 1.01:
            upper_m_N_vals[index] = m_N
            converged = True
            
            #Check if it converged on a max of min
            if m_N_limits[0] == starting_m_N_limits[0] or m_N_limits[1]== starting_m_N_limits[1]:
                outside_bound_indeces = np.append(outside_bound_indices,index)
        
    index += 1
#Cut out values that did not converge
try:
    cut = min(outside_bound_indices)
except:
    cut = len(lower_m_N_vals)

upper_m_N_vals = upper_m_N_vals[0:cut]
upper_d_vals = upper_d_vals[0:cut]

d_vals = np.append(lower_d_vals, upper_d_vals)
m_N_vals = np.append(lower_m_N_vals, upper_m_N_vals)

fig = plt.figure()
plt.plot(m_N_vals, d_vals)
plt.xlabel('$\mathrm{m_N}$ (GeV)')
plt.ylabel('d ($\mathrm{MeV^{-1}}$)')
plt.xscale('log')
plt.yscale('log')
