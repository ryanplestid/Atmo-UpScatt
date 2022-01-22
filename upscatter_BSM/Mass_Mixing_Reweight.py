'''
Module made to reweight a Monte Carlo integration for the HNL mass-mixing portal with a new
    HNL mass and coupling strength.  It requires the filename from a previous Monte Carlo 
    integration to run.
'''

import numpy as np
from numpy import sin, cos, pi
from numpy import random as rand 
import scipy as sp
from scipy import special as spc
import matplotlib as mpl #Package to help with plotting
from matplotlib import pyplot as plt
from matplotlib import ticker, cm
import pickle
plt.style.use(['science','ieee'])

#Import modules that we made
#import SampleEvents
import Mass_Mixing_Model
import atmoNuIntensity
import earthComp
import DetectorModule
import oscillations
import MassMixingMainIntegration
from MassMixingMainIntegration import MassMixMonteCarlo
import MassMixingDetectorModule
from reweightModule import Update_Fluxes

def ReIntegrate(filename,U,m_N,scattering_channel, V_det = None, flavors = ['E','EBar','Mu','MuBar','Tau','TauBar'], Flav = 'e'):
    '''
    Re-integrates to find the rate given a file with meta data and event objects
    
    args:
        filename: Name of the file that contains the necessary information (str)
        U = Mass Mixing coupling
        m_N = HNL mass (GeV)
        flavors: Flavors of neutrinos for which we are interested in the interactions
        Flav: Flavor of the HNL coupling, dictates the type of interactions that can happen
    
    returns:
        tot_integral: Rate of sub-GeV HNL events in the detector in s^{-1} (float)
        rate_error: Uncertainty in the rate in s^{-1} (float)
        E_ee_midpoints: Midpoints of the bins in lepton pair energy in GeV (array)
        E_ee_rates: Rate in each bin of lepton pair energy in s^{-1} (array)

    '''
    #Perform Oscillations
    Update_Fluxes(filename, 0.1)
    
    #Load Data
    Sim_Dict = pickle.load(open(filename,"rb"))
    Meta_Data = Sim_Dict["MetaData"]
    Event_list = Sim_Dict["EventData"]
    
    Y = Meta_Data['Detector Location (r_Earth=1)']
    R_Earth = 1 # R_Earth = 1 #
    R_Earth_cm = 6378.1 * 1000* 100    #Radius of the Earth (cm)
    
    Energy_range = Meta_Data['Energy Limits'][1] - Meta_Data['Energy Limits'][0] #GeV
    Theta_min= Meta_Data['Theta min']
    Theta_max = Meta_Data['Theta max']
    
    Earth_Volume = (4*pi/3)*R_Earth**3 #cm^3
    
    if V_det == None:
        V_det = Meta_Data["Detector Volume (cm^3)"]
    
    A_perp = (V_det)**(2/3) #cm^2
    l_det = (V_det)**(1/3) / R_Earth_cm # units of R_Earth = 1
    E_nus = np.zeros(len(Event_list))
    E_HNLs = np.zeros(len(Event_list))
    dRs = np.zeros(len(Event_list))
    N_Fluxes = np.zeros(len(Event_list))
    X_vect_vals = np.zeros((len(Event_list),3))
    N_sigma_cos_Thetas = np.zeros(len(Event_list))
    cos_Thetas = np.zeros(len(Event_list))
    
    tot_integral = 0
    trial = 0
    for event in Event_list:
        Cos_Theta_range = cos(Theta_min) - cos(Theta_max[trial])
        E_nus[trial] = event.nu_Energy
        E_HNLs[trial] = event.N_Energy
        X_vect_vals[trial,:] = event.Interact_Pos
        
        #N_lambda = dipoleModel.decay_length(d,m_N, event.N_Energy)
        N_lambda = Mass_Mixing_Model.decay_length(Flav, U, m_N, np.array([E_HNLs[trial]]))
            
        P_dec_A_Perp = (np.exp(-event.Dist_to_det/N_lambda) * (1-np.exp(-l_det/N_lambda))
                        *A_perp)
        
        #Find the total flux for desired flavors
        tot_flux = 0
        for flavor in flavors:
            if flavor == 'E':
                tot_flux += event.NuEFlux
            elif flavor == 'EBar':
                tot_flux += event.NuEbarFlux
            elif flavor == 'Mu':
                tot_flux += event.NuMuFlux
            elif flavor == 'MuBar':
                tot_flux += event.NuMubarFlux
            elif flavor == 'Tau':
                tot_flux += event.NuTauFlux
            elif flavor == 'TauBar':
                tot_flux += event.NuTaubarFlux
            else:
                print(flavor)
        
        weights = event.E_weight * event.V_weight * event.Theta_weight
        tot_delta = R_Earth_cm* Energy_range * Cos_Theta_range * Earth_Volume * weights / len(Event_list)
        N_d_sigma_d_cos_Theta = Mass_Mixing_Model.N_Cross_Sec_from_radii(U,m_N,[event.nu_Energy],
                                                                         [event.cos_Theta],[event.r],
                                                                         scattering_channel)
        
        N_d_sigma_d_cos_Theta = float(N_d_sigma_d_cos_Theta)
        N_sigma_cos_Thetas[trial] = N_d_sigma_d_cos_Theta
        cos_Thetas[trial] = event.cos_Theta
        
        visible_frac = Mass_Mixing_Model.Gamma_partial(Flav, m_N, U, final_state = 'nu e e')\
                /Mass_Mixing_Model.Gamma_tot(Flav,m_N,U)
        
        dRs[trial] = (N_d_sigma_d_cos_Theta *4*pi*tot_flux 
                         * (P_dec_A_Perp)/(4*pi*event.Dist_to_det**2) * tot_delta 
                         * visible_frac)
        

        N_Fluxes[trial] = (N_d_sigma_d_cos_Theta *4*pi*tot_flux 
                         * np.exp(-event.Dist_to_det/N_lambda) /(4*pi*event.Dist_to_det**2) * tot_delta)
        
        trial += 1

    
    #find energies for the e+ e- pair
    E_ee_bounds = np.linspace(0.03, 1.33 ,11)
    E_ee_midpoints = (E_ee_bounds[:10] + E_ee_bounds[1:])/2
    E_ee_rates = np.zeros(len(E_ee_midpoints))
    rate = 0
    high_E_rate = 0
    
    #Find energy of the lepton pair in the lab frame
    decay_trials = 10
    for i in range(decay_trials):
        E_decay_nu_lab = MassMixingDetectorModule.Sample_Decay_E_nu_Lab(m_N, E_HNLs, 'Vector')
        E_ees = E_HNLs - E_decay_nu_lab
        
        for event_index in range(len(E_ees)):
            E_ee = E_ees[event_index]
            dR = dRs[event_index] / decay_trials
            #See if lepton pair energy makes cut
            if E_ee > 0.030 and E_ee < 1.33:
                rate += dR
                
                for energy_index in range(len(E_ee_rates)):
                    if E_ee_bounds[energy_index]<E_ee and E_ee < E_ee_bounds[energy_index+1]:
                        E_ee_rates[energy_index] += dR
            if E_ee > 1.33:
                high_E_rate += dR
        
    
    
    rate_error = np.std(dRs)*np.sqrt(len(dRs))
    
    print('% Error', (rate_error/rate)*100)
    
    return(rate,rate_error, E_ee_midpoints, E_ee_rates)

m_N_vals = np.logspace(-2,-0.5,8)
U2_vals = np.logspace(-2,0, 8)

'''
m_N = 0.03
U2 = 0.3

least_greater_m_N_index = int(np.where((m_N_vals - m_N) > 0)[0][0]) 
least_greater_m_N = m_N_vals[least_greater_m_N_index]
        
nearest_U_index = int(np.where(abs(U2_vals - U2) == min(abs(U2_vals - U2)))[0])
nearest_U2 = U2_vals[nearest_U_index]
        
filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'nucleon' +" exp.events"


rate, rate_error, E_ee_midpoints, E_ee_rates, high_E_rate = ReIntegrate(filename,np.sqrt(U2),m_N,'nucleon',Flav = 'tau',flavors=['Tau','TauBar'])


fig = plt.figure(figsize = (4,3))
plt.bar(E_ee_midpoints, E_ee_rates * 86400*5326, width = .13,alpha = 0.5)
plt.xlabel('$E_{e^{+} e^{-}}$ (GeV)')
plt.ylabel('Events / 5326 Days')
plt.text(.9,50,'Nucleon')
plt.text(.9,40,'$\mathrm{m_N}$ = ' + str(m_N) +" GeV",fontsize = 6)
plt.text(.9,30, "$\mathrm{|U|^2}$ = %.3g"%U2,fontsize = 6)
plt.xticks([0.03,.29,.55,.81,1.07,1.33])
print('High E events', high_E_rate*86400*5326)
'''

'''
Rates = np.zeros((len(m_N_vals),len(U2_vals)))

for m_N_index in range(len(m_N_vals)):
    m_N = m_N_vals[m_N_index]
    
    
    for U_index in range(len(U2_vals)):
        U = np.sqrt(U2_vals[U_index])
        
        print('m_N', m_N)
        print('U^2', U**2)
        filename, co_Decays = MassMixMonteCarlo([m_N],[U], int(5e4),'coherent',flav ='tau')
        filename, nu_Decays = MassMixMonteCarlo([m_N],[U], int(5e4),'nucleon',flav ='tau')
        filename, DIS_Decays = MassMixMonteCarlo([m_N],[U], int(5e4),'DIS',flav ='tau')
        
        coherent_filename = "mN_%.3g" %m_N+"_U_%.3g"%U+ 'coherent'+" exp.events"
        nucleon_filename = "mN_%.3g" %m_N+"_U_%.3g"%U+ 'nucleon'+" exp.events"
        DIS_filename = "mN_%.3g" %m_N+"_U_%.3g"%U+ 'DIS'+" exp.events"
        
        co_Decays, co_Error,ee_mid,ee_rate = ReIntegrate(coherent_filename,U,m_N,'coherent',Flav = 'tau')
        nu_Decays, nu_Error,ee_mid,ee_rate = ReIntegrate(nucleon_filename,U,m_N,'nucleon',Flav = 'tau')
        DIS_Decays, DIS_Error,ee_mid,ee_rate = ReIntegrate(DIS_filename,U,m_N,'DIS',Flav = 'tau')
        
        print('coherent rate', co_Decays)
        print('nucleon rate', nu_Decays)
        print('DIS rate',DIS_Decays)
        
        Rates[m_N_index,U_index] =float(nu_Decays) + float(co_Decays) + float(DIS_Decays)
        print('total rate', Rates[m_N_index, U_index])
        print(' ')
        

figk, ax = plt.subplots(1,1)
#levels = np.logspace(-6,4,6)
levels = np.array([0,1,30,100,300])
cp = ax.contourf(m_N_vals,U2_vals, np.transpose(Rates) * 86400 * 5326, levels, cmap = cm.viridis)
figk.colorbar(cp)
plt.ylabel('$U_{\\tau}^{2}$ ', fontsize = 14)
plt.yscale('log')
plt.xlabel('$M_N$ (GeV)', fontsize = 14)
plt.xscale('log')
#plt.suptitle('HNL Decays, tau only', fontsize = 6)
plt.title('Super-K, 5326 days, all flux',fontsize = 6)



figk, ax = plt.subplots(1,1)
#levels = np.logspace(-6,4,6)
levels = np.array([0,1,30,100,300])
cp = ax.contourf(m_N_vals, U2_vals, np.transpose(Rates)/10 * 86400 * 5326, levels, cmap = cm.viridis)
figk.colorbar(cp)
plt.ylabel('$U_{\\tau}^{2}$ ', fontsize = 14)
plt.yscale('log')
plt.xlabel('$M_N$ (GeV)', fontsize = 14)
plt.xscale('log')
#plt.suptitle('HNL Decays, tau only', fontsize = 6)
plt.title('Super-K, 5326 days, 10 percent all flux', fontsize = 6)
'''

'''
m_N = 0.03
U2 = 0.06
U = np.sqrt(U2)

least_greater_m_N_index = int(np.where((m_N_vals - m_N) > 0)[0][0]) 
least_greater_m_N = m_N_vals[least_greater_m_N_index]
        
nearest_U_index = int(np.where(abs(U2_vals - U2) == min(abs(U2_vals - U2)))[0])
nearest_U2 = U2_vals[nearest_U_index]
        
Co_filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'coherent' +" exp.events"
Nuc_filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'nucleon' +" exp.events"
DIS_filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'DIS' +" exp.events"

Co_rate, Co_rate_error, E_ee_midpoints, E_ee_rates \
    = ReIntegrate(Co_filename,U,m_N,'coherent',Flav = 'tau',flavors = ['Tau','TauBar'])
Nuc_rate, Nuc_rate_error, E_ee_midpoints, E_ee_rates \
    = ReIntegrate(Nuc_filename,U,m_N,'nucleon',Flav = 'tau',flavors = ['Tau','TauBar'])
DIS_rate, DIS_rate_error, E_ee_midpoints, E_ee_rates \
    = ReIntegrate(DIS_filename,U,m_N,'DIS',Flav = 'tau',flavors = ['Tau','TauBar'])
    
print('Co %', Co_rate/(Co_rate+Nuc_rate+DIS_rate) * 100)
print('Nuc %', Nuc_rate/(Co_rate+Nuc_rate+DIS_rate)* 100)
print('DIS %', DIS_rate/(Co_rate+Nuc_rate+DIS_rate) * 100)
'''


Detector_Volume = 22500 * 1e6 #cm^2
desired_excess = 10
'''
limits_mn_vals = np.logspace(-0.8,-0.3,3)
limits_U2_tau_vals = np.zeros(3)
i = -1

for m_N in limits_mn_vals:
    i += 1
    print('m_N', m_N)
    U2_bounds = [10**(-3),10**(-0.5)]
    repeat = True
    
    while repeat:
        U2 = np.sqrt(U2_bounds[0]*U2_bounds[1])
        print('U^2', U2)
        U = np.sqrt(U2)
        
        least_greater_m_N_index = int(np.where((m_N_vals - m_N) > 0)[0][0]) 
        least_greater_m_N = m_N_vals[least_greater_m_N_index]
                
        nearest_U_index = int(np.where(abs(U2_vals - U2) == min(abs(U2_vals - U2)))[0])
        nearest_U2 = U2_vals[nearest_U_index]
                
        Co_filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'coherent' +" exp.events"
        Nuc_filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'nucleon' +" exp.events"
        DIS_filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'DIS' +" exp.events"
        
        Co_rate, Co_rate_error, E_ee_midpoints, E_ee_rates \
            = ReIntegrate(Co_filename,U,m_N,'coherent',Flav = 'tau',flavors = ['Tau','TauBar'])
        Nuc_rate, Nuc_rate_error, E_ee_midpoints, E_ee_rates \
            = ReIntegrate(Nuc_filename,U,m_N,'nucleon',Flav = 'tau',flavors = ['Tau','TauBar'])
        DIS_rate, DIS_rate_error, E_ee_midpoints, E_ee_rates \
            = ReIntegrate(DIS_filename,U,m_N,'DIS',Flav = 'tau',flavors = ['Tau','TauBar'])
            
        events = (Co_rate + Nuc_rate*2 + DIS_rate) * 86400 * 5326
        print('DIS %', (DIS_rate/(Co_rate+Nuc_rate+DIS_rate)) * 100)
        
        if abs(events - desired_excess) < 1:
            limits_U2_tau_vals[i] = U2
            print('U^2 good')
            repeat = False
        elif events > desired_excess:
            U2_bounds[1] = U2
            print('U^2 too high')
        elif events < desired_excess:
            U2_bounds[0] = U2
            print('U^2 too low')
        
        if U2_bounds[1]/U2_bounds[0] < 1.01:
            limits_U2_tau_vals[i] = U2
            repeat = False

#fig = plt.figure()
plt.plot(limits_mn_vals, limits_U2_tau_vals)
plt.xlabel('$\mathrm{m_N}$ GeV')
plt.ylabel('$\mathrm{|U_{\\tau N}|^2}$')
plt.xscale('log')
plt.yscale('log')
'''

limits_mn_vals = np.zeros(8)
limits_U2_tau_vals = np.logspace(-1.7,0,8)
i = -1

for U2 in limits_U2_tau_vals:
    i += 1
    print('U2', U2)
    mn_bounds = [0.02,0.2]
    repeat = True
    
    while repeat:
        m_N = np.sqrt(mn_bounds[0]*mn_bounds[1])
        print('m_N', m_N)
        U = np.sqrt(U2)
        
        least_greater_m_N_index = int(np.where((m_N_vals - m_N) > 0)[0][0]) 
        least_greater_m_N = m_N_vals[least_greater_m_N_index]
                
        nearest_U_index = int(np.where(abs(U2_vals - U2) == min(abs(U2_vals - U2)))[0])
        nearest_U2 = U2_vals[nearest_U_index]
                
        Co_filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'coherent' +" exp.events"
        Nuc_filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'nucleon' +" exp.events"
        DIS_filename = "mN_%.3g" %least_greater_m_N+"_U_%.3g"%np.sqrt(nearest_U2)+'DIS' +" exp.events"
        
        Co_rate, Co_rate_error, E_ee_midpoints, E_ee_rates \
            = ReIntegrate(Co_filename,U,m_N,'coherent',Flav = 'tau',flavors = ['Tau','TauBar'])
        Nuc_rate, Nuc_rate_error, E_ee_midpoints, E_ee_rates \
            = ReIntegrate(Nuc_filename,U,m_N,'nucleon',Flav = 'tau',flavors = ['Tau','TauBar'])
        DIS_rate, DIS_rate_error, E_ee_midpoints, E_ee_rates \
            = ReIntegrate(DIS_filename,U,m_N,'DIS',Flav = 'tau',flavors = ['Tau','TauBar'])
            
        events = (Co_rate + Nuc_rate + DIS_rate) * 86400 * 5326
        print('DIS %', (DIS_rate/(Co_rate+Nuc_rate+DIS_rate)) * 100)
        
        if abs(events - desired_excess) < 1:
            limits_mn_vals[i] = m_N
            print('m_N good')
            repeat = False
        elif events > desired_excess:
            mn_bounds[0] = m_N
            print('too high')
        elif events < desired_excess:
            mn_bounds[1] = m_N
            print('too low')
        
        if mn_bounds[1]/mn_bounds[0] < 1.01:
            limits_mn_vals[i] = m_N
            repeat = False

'''
plt.plot(limits_mn_vals,limits_U2_tau_vals)
plt.xlabel('$\mathrm{m_N}$ GeV')
plt.ylabel('$\mathrm{|U_{\\tau N}|^2}$')  
plt.xscale('log')
plt.yscale('log')
'''

fig = plt.figure()
plt.plot(limits_mn_vals, limits_U2_tau_vals)
plt.xlabel('$\mathrm{m_N}$ GeV')
plt.ylabel('$\mathrm{|U_{\\tau N}|^2}$')  
plt.xscale('log')
plt.yscale('log')

'''
m_N = 0.02
U = np.sqrt(0.1)

filename, nu_Decays = MassMixMonteCarlo([m_N],[U], int(2e4),'DIS',flav ='tau')
#scattering_channel = 'nucleon'
#filename = "mN_%.3g" %m_N+"_U_%.3g"%U+ str(scattering_channel)+" exp.events"
rate, rate_error = ReIntegrate(filename,U,m_N,'DIS',Flav = 'tau')

print('m_N', m_N, ' U^2 ', U**2)
print('nu_Decays', nu_Decays*86400*4438)
print('Decays in 4438 Days', rate*86400*4438)
'''