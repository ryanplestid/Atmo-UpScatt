#Initialization

'''
Module to compute the form factors for nuclei to determine
the contributions of coherent and incoherent scattering.

functions:
    Calculate_Fermi_3p_FF2: Calculates an array of the form-factor squared
        at specified q values, computed via a 3-parameter Fermi model
    Find_Helm_Parameters: Perform a fit of Helm form-factor parameters
        to the array of form factors computed using the 3-parameter
        Fermi model
    Helm_FF2: Calculate the form-factor squared at a specified q value
        in the Helm model
dictionaries:
    Element_Dict: Earth elements with their atomic numbers and molar
        mass for their most common isotope.
    Fermi_Dict: 3-parameter Fermi values for common Earth isotopes
    Helm_Dict: Helm parameters for common Earth isotopes
'''


import math
import numpy as np #Package for array functions
import scipy as sp
from numpy import pi as pi
from scipy import special as spc
from numpy import random as rand
from numpy import sin, cos



def main():
    print("Hello World")
    return 0

Element_Dict = dict({"O": "8 16",
                     "Mn": "12 24",
                     "Al": "13 27",
                     "Si": "14 28",
                     "K": "19 39",
                     "Ca": "20 40",
                     "Ti": "22 48",
                     "Cr": "24 52",
                     "Mn": "25 55",
                     "Fe": "26 56",
                     "Ni": "28 58"})

# Create a global dictionary of Fermi parameters
# Enter two numbers(Z A) as a string, separated by a space
# Form Factors found from De Vries, De Jager, and De Vries (1987)
# Sodium and Sulfur does not have a Fermi Form Factor
Fermi_Dict = dict({"8 16":dict({"c":2.608, "z": 0.513, "w": -0.051}),
                   "12 24":dict({"c":2.98,"z":0.551, "w": 0}),
                   "13 27":dict({"c":2.84, "z":0.569, "w":0}),
                   "14 28":dict({"c":3.340, "z": 0.580, "w": -0.233}),
                   "19 39":dict({"c":3.408, "z": 0.585, "w": -0.201}),
                   "20 40":dict({"c":3.766, "z": 0.586, "w":-0.161}),
                   "22 48":dict({"c":3.843, "z": 0.588, "w": 0}),
                   "24 52":dict({"c":4.01, "z":0.497,"w":0}),
                   "25 55":dict({"c":3.89, "z":0.567,"w":0}),
                   #"26 54":dict({"c":4.075, "z":0.506, "w":0}),
                   "26 56":dict({"c":4.106, "z": 0.519, "w":0}),
                   "28 58":dict({"c":4.3092, "z": 0.5169,"w":-0.1308}),
                   #"28 60":dict({"c":4.4891,"z":0.5369,"w":-0.2668})
                  })




def Calculate_Fermi_3P_FF2(Fermi_c, Fermi_z, Fermi_w, q_vals_fm):
    '''
    This calculates the value of the 3 parameter Fermi Form Factor squared
    at specified q values.
    
    args:
        Fermi_c: Fermi c charge distribution parameter in fm (float)
        Fermi_z: Fermi z charge distribution parameter in fm (float)
        Fermi_w: Fermi w charge distribution parameter (float)
        q_vals_fm: Momentum transfer values (in fm) at which we will evaluate
                the form factor (array, floats)
    
    returns:
        Fermi_FF2: array of the Fermi Form Factor squared at the specified
                q values
    
    actions:
        Calculates Form Factor squared and returns it
    '''
    #Constants
    i = 1j
    #e = .303 #Elementary charge, Natural Units
    
    #Arrays and step sizes for the integration
    r_vals = np.linspace(0, 5*Fermi_c, 1000) #fm
    cos_theta_vals = np.linspace(-1,1,50)
    dr = r_vals[1] - r_vals[0]
    d_cos_theta = cos_theta_vals[1] - cos_theta_vals[0]

    #Initialize the values of F^2(q) as 0
    Fermi_FF2 = np.zeros(len(q_vals_fm))
    
    #Calculate F^2(q) at specified values of q
    q_index = 0
    for q in q_vals_fm:
        
        #Perform the integral over r and cos(theta) to get F(q)
        
        fourier_vals = (np.exp(i*q*np.outer(cos_theta_vals,r_vals))* 2*pi * r_vals**2 
                        * (1+Fermi_w *r_vals**2/Fermi_c**2) /(1+np.exp((r_vals-Fermi_c)/Fermi_z)) *dr*d_cos_theta)
        fourier_int = sum(sum(fourier_vals))
        
        #Square the integrated value to get F^2(q)
        Fermi_FF2[q_index] = np.real(fourier_int * np.conj(fourier_int))
        q_index += 1
    
    #Normalize F^2(q)
    Fermi_FF2 = Fermi_FF2 / max(Fermi_FF2)
    return(Fermi_FF2)

def Find_Helm_Parameters(Zed,Fermi_c,Fermi_z, q_vals, Fermi_w = 0):
    '''
    This function finds the best Helm Form Factor parameters from
    the Fermi charge distribution parameters.
    
    args:
        Zed: Atomic number of the nuclei (int)
        Fermi_c: Fermi c charge distribution parameter in fm (float)
        Fermi_z: Fermi z charge distribution parameter in fm (float)
        q_vals: Momentum transfer values (in fm) at which we will evaluate
                and compare the form factors (array, floats)
        Fermi_w: Fermi w charge distribution parameter (float). Set to 0 if none is specified
    
    returns:
        R1: Effective radius for Helm Form Factor in fm (float)
        s: Skin thickness for Helm Form Factor in fm(float)
    
    actions:
        Guesses an initial R1 and s, then iteratively impoves to fit to
        the Fermi Form Factor using a Gradient Ascent Method for the 
        integral of the difference between the form factors squared.
    '''
    #Initial guess
    Ra = 1.23 * (2*Zed)**(1/3) - 0.6 #fm
    r0 = 0.52 #fm
    s = Fermi_z #fm
    delta_Ra = (1.23 * (2*Zed)**(1/3) - 0.6)/5
    delta_r0 = (0.52)/5
    delta_s = Fermi_z/5
    
    #Calculate the Fermi Form factor at the specified values of q
    Fermi_FF2 = Calculate_Fermi_3P_FF2(Fermi_c,Fermi_z,Fermi_w,q_vals) #Form factors from fermi distribution

    '''
    fig = plt.figure(figsize = (8,6))
    plt.plot(q_vals, Fermi_FF2, label = 'Fermi')
    print('Fermi done')
    '''
    #Initialize the Helm Form factor as zeros
    Helm_FF2 = np.zeros(len(Fermi_FF2))  #Helm form factors
    
    #Perform a set number of trials to find the best fit parameters
    for trial in range(401):
        #Calculate the Helm Form Factor squared with the parameters as is
        
        R1 = np.sqrt(Ra**2+(7/3)*pi**2*r0**2 -5*s**2)
        Helm_FF2 = ((3*spc.spherical_jn(1,q_vals*R1))/(q_vals*R1))**2 *np.exp((q_vals*s)**2)
        '''
        if trial%200 == 0:
            plt.plot(q_vals, Helm_FF2, label = "Helm trial = "+str(trial))
            plt.legend()
        '''
        
        #Find the difference between the Helm and Fermi form factors
        #  with the Helm parameters as is
        initial_difference = sum((Helm_FF2 - Fermi_FF2)**2)
        
        #Increase our R_a value and find the new difference between the form factors
        Ra_p = Ra + 0.1*delta_Ra
        R1 = np.sqrt(Ra_p**2+(7/3)*pi**2*r0**2 -5*s**2)
        Helm_FF2 = ((3*spc.spherical_jn(1,q_vals*R1))/(q_vals*R1))**2 *np.exp((q_vals*s)**2)
        
        Ra_difference = sum((Helm_FF2 - Fermi_FF2)**2)
        
        #Increase our s value and find the new difference between the form factors
        s_p = s+ delta_s
        R1 = np.sqrt(Ra**2+(7/3)*pi**2*r0**2 -5*s_p**2)
        Helm_FF2 = ((3*spc.spherical_jn(1,q_vals*R1))/(q_vals*R1))**2 *np.exp((q_vals*s_p)**2)
        
        s_difference = sum((Helm_FF2 - Fermi_FF2)**2)
        
        #Increase our r_0 value and find the new difference between the form factors
        r0_p = r0+ delta_r0
        R1 = np.sqrt(Ra**2+(7/3)*pi**2*r0_p**2 -5*s**2)
        Helm_FF2 = ((3*spc.spherical_jn(1,q_vals*R1))/(q_vals*R1))**2 *np.exp((q_vals*s)**2)
        
        r0_difference = sum((Helm_FF2 - Fermi_FF2)**2)
    
        #See how the difference between the Fermi and Helm form factors
        #  depended on our Helm parameters
        dD_dRa = Ra_difference - initial_difference
        dD_ds = s_difference - initial_difference
        dD_dr0 = r0_difference - initial_difference
        
        #Alter Helm parameters so as to decrease the difference between
        # the Fermi and Helm form factors
        Ra = Ra - dD_dRa *delta_Ra
        s = s - dD_ds * delta_s
        r0 = r0 - dD_dr0 *delta_r0

        #Calculate R1 with the improved parameters.
        R1 = np.sqrt(Ra**2+(7/3)*pi**2*r0**2 -5*s**2)
    return(R1,s)


#Create a dictionary for the Helm Parameters by fitting the fermi parameters
#Enter two numbers (Z A) as a string, separted by a space
#Returns another dictionary for R1 and s in fm
Helm_Dict= dict({})
q_vals = np.linspace(.01,2,100) #fm^-1
for key in Fermi_Dict.keys():
    #Get the Fermi parameters
    element = Fermi_Dict[key]
    Zed_str,A_str = key.split(" ")
    Fermi_c,Fermi_z,Fermi_w = element["c"], element["z"], element["w"]
    #Fit the Helm Parameters
    R1,s = Find_Helm_Parameters(int(Zed_str),Fermi_c,Fermi_z,q_vals,Fermi_w)
    Helm_Dict[key] =dict({"R1":R1,"s":s})

#Calculate the Helm Form Factor
def Helm_FF2(q_MeV,R1,s):
    '''
    Calculates the Helm Form Factor at a desired momentum transfer
    given the effective radius and the skin thickness.
    
    args:
        q_MeV: Transfered momentum in MeV (float)
        R1: Effecitive radius of the Helm Form Factor in fm (float)
        s: Skin thickness of the Helm Form Factor in fm (float)
        
    returns:
        FF2: Helm Form Factor squared at specified momentum transfer
        
    actions:
        Calculate the Helm Form Factor squared
    '''
    q_fm = 1/197.3 * q_MeV #fm^-1
    FF2 = (3*spc.spherical_jn(1,q_fm*R1)/(q_fm*R1))**2 * np.exp(-(q_fm*s)**2)

    return(FF2)


if __name__ == "__main__":
    main()
