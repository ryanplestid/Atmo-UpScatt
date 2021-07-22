def main():
    print("Hello World")
    return 0

import numpy as np
from numpy import sin, cos, pi
from numpy import random as rand 
import scipy as sp
from scipy import special as spc

#Import modules that we made
import SampleEvents
import dipoleModel
import atmoNuIntensity
import earthComp
import DetectorModule

#Parameters
E_min,E_max = 0.1, 1e4 #min/max Neutrino Energy, GeV
power_law = 2 #Guess at energy dependence of atm flux
flux_name= 'H3a_SIBYLL23C'

R_Earth = 6378.1 * 1000* 100    #Radius of the Earth (cm)
c = 10 #R_max = c*lambda
Y = np.array([0,0,R_Earth])     #Location of our detector
V_det = 300 * (1e6)         #Volume of detector in cm^3

epsilon = 1e-7 #1 - cos(Theta_min)
Theta_max = pi

m_N = 0.01 #HNL mass, GeV
d = 1e-9 #Dipole coupling (MeV^-1)
num_Events = int(1e5)

#Sample events
Energies = SampleEvents.Sample_Neutrino_Energies(num_Events,E_min,E_max,power_law)
lambdas = dipoleModel.decay_length(d,m_N,Energies)
R_maxs = c*lambdas

X_vect_vals = SampleEvents.Sample_Interaction_Locations(num_Events, Y, R_maxs)
Radii = np.sqrt(X_vect_vals[:,0]**2 +X_vect_vals[:,1]**2 + X_vect_vals[:,2]**2) #cm
rs = Radii/R_Earth #normalized radii
cos_Thetas = SampleEvents.Sample_cos_Theta(num_Events, epsilon, Theta_max)
W_vect_vals = SampleEvents.Sample_Neutrino_Entry_Position(X_vect_vals,Y,cos_Thetas)

#Calculate Flux
cos_zeniths = atmoNuIntensity.Calc_cos_zeniths(X_vect_vals,W_vect_vals)
flux = atmoNuIntensity.calc_fluxes(Energies,cos_zeniths, flux_name) #Flux in GeV^-1 cm^-2 sr^-1 s^-1

#Calculate cross section x number density
N_d_sigma_d_cos_Thetas = dipoleModel.N_Cross_Sec_from_radii(d,m_N,Energies,cos_Thetas,rs)

#Prob to decay in detector * Perpendicular Area
Y_minus_X_mag = np.sqrt((Y[0] - X_vect_vals[:,0])**2 + (Y[1] - X_vect_vals[:,1])**2 + (Y[2] - X_vect_vals[:,2])**2)
P_decs_A_Perp = np.exp(-Y_minus_X_mag/lambdas) * V_det/lambdas

#Calculate weights for events
E_Range = E_max - E_min
cos_Theta_range = (1-epsilon) - cos(Theta_max)
Volume = 4*pi/3 * R_Earth ** 3

w_E = SampleEvents.weight_Energy(Energies, E_min, E_max, power_law)
w_V = SampleEvents.weight_positions(Y,R_maxs)
w_Theta = SampleEvents.weight_cos_Theta(cos_Thetas,epsilon, Theta_max)
tot_weight = (w_E * w_V * w_Theta)

tot_delta = E_Range * cos_Theta_range * Volume * tot_weight/num_Events

#Calculate how much each event contributes to the rate

dR = N_d_sigma_d_cos_Thetas * 4 * pi * flux * (P_decs_A_Perp)/(4*pi * Y_minus_X_mag**2) * tot_delta
Rate = sum(dR)
print('Rate', Rate)
print('decays in 12 years', Rate * 86400 * 365 *12)

if __name__ == "__main__":
    main()
