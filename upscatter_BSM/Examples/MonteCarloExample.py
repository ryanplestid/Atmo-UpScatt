'''
Example showing how to run the Monte Carlo simulation
    and then use the ReIntegrate function to recalculate
    the rate of HNL events
    
We also show how to see the angular and energetic data
    from the ReIntegrate function
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

#Specify parameters for Monte Carlo Simulation
MC_m_N = 0.03 #GeV
MC_d = 2*1e-10 #MeV^{-1}
num_events = int(2e4)

filename, Decays = MainIntegration.MonteCarlo([MC_m_N],[MC_d],num_events)

#filename contains the events used in the Monte Carlo
#Decays gives the number of HNL events per second, with Super-K
#   as the specified volume

#ReIntegrate
#Specify parameters for ReIntegration
Re_m_N = 0.04 #GeV
Re_d = 2.5 * 1e-10 #MeV^{-1}
V_det = 22500 * 1e6 #Detector volume (cm^3)
alpha_decay = 0 #0 for Majoranna HNL, [-1,1] for Dirac
flavors = ['E', 'EBar','Mu', 'MuBar', 'Tau', 'TauBar'] 
#neutrino flavors coupled to HNL
total_time = 10 * 365 * 86400 #total live time (in s)

#Perform oscillations on the flux
#We do not oscillate the events that contribute least to the flus
#   The sum contributions of these non-oscillated events to the
#   total flux is threshold*(total rate)
threshold = 0.05

reweightModule.Update_Fluxes(filename,threshold) #Performs oscillations
#Oscillations are the slowest part of this process.
#If you are working in a flavor ind model, we suggest
#   you skip oscillations

Rate,rate_error,cos_phi_midpoints, angular_rates, \
    E_gamma_midpoints,E_gamma_rates,high_E_rate \
        =reweightModule.ReIntegrate(filename, Re_d, Re_m_N, alpha_decay, V_det, flavors)


print('# of Decays with 30 MeV < E_photon < 1.33 GeV: ', Rate*total_time)
print('Uncertainty in # of Decays with 30 MeV < E_photon < 1.33 GeV: ',
      rate_error * total_time)

print('# of Decays with E_photon > 1.33 GeV: ', high_E_rate*total_time)

#Make distributions of angles and Energies

fig = plt.figure(figsize = (4,3))
plt.bar(E_gamma_midpoints, E_gamma_rates *total_time, width = .13,alpha = 0.5)
plt.xlabel('$E_{\gamma}$ (GeV)')
plt.ylabel('Visible HNL Decays in '+str(total_time)+'s')
plt.title('$\mathrm{m_N}$ =' + str(Re_m_N) +" GeV ; d = "+str(Re_d)+" $\mathrm{MeV^{-1}}$")
plt.xticks([0.03,.29,.55,.81,1.07,1.33])


fig = plt.figure(figsize = (4,3))
plt.bar(cos_phi_midpoints, angular_rates * total_time, width = .2,alpha = 0.5)
plt.xlabel('$\cos(\phi_{\mathrm{det}})$')
plt.ylabel('Visible HNL Decays in '+str(total_time)+'s')
plt.title('$\mathrm{m_N}$ =' + str(Re_m_N) +" GeV ; d = "+str(Re_d)+" $\mathrm{MeV^{-1}}$")
