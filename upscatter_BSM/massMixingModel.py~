def main():
    print("Hello World")
    return 0

#Initialization
import numpy as np #Package for array functions
import formFactorFit
import earthComp
from numpy import random as rand
from numpy import sin, cos

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


def decay_length(d,mn,En):
    '''
    Find the characteristic decay length of the lepton
    
    args:
        d: dipole coupling constant in MeV^-1 (float)
        mn: mass of the lepton in GeV (float)
        En: Energy of the lepton in GeV (float or array of floats)
    
    returns:
        Lambda: Characteristic decay length in units of R_earth=1 (float or array of floats, equal to the number of energies)
        
    action:
        Calculates characteristic decay length according to Plestid, 2021
    '''
    #Set decay length to 0 if mass is greater than energy
    '''
    if mn >= En:
        Lambda = 0
        return(Lambda)
    '''
    
    #R_Earth = 6378.1 * 1000* 100    #Radius of the Earth (cm)
    R_Earth = 1
    mn_MeV = mn*1000  #Convert mass to MeV
    En_MeV = En*1000  #Convert energy to MeV
    Lambda = (R_Earth * (1.97e-9/d)**2 *(1/mn_MeV)**4 * (En_MeV/10) 
              * np.sqrt((1-mn**2/En**2)/0.99) )#Decay lengths (cm)
    
    Lambda = Lambda * np.heaviside(En - mn,0)
    
    return(Lambda)


def d_sigma_d_cos_Theta_coherent(d,mn,En,cos_Theta,Zed):
    '''
    Determine the differential cross section for a coherent scattering at a specified angle
    
    args:
        d: dipole coupling constant in MeV^-1 (float)
        mn: mass of the lepton in GeV (float)
        En: Energy of the lepton in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        Zed: Atomic number (int)
        
    returns:
        d_sigma_d_cos_Theta: differential upscattering cross section in cm^2 
                            (float, same size as En)
    
    actions:
        Computes the differential cross section in terms of MeV^{-2}, then
        converts it to cm^2
    '''
    #Set differential cross section to 0 if the mass is greater than the energy
    if mn >= En:
        d_sigma_d_cos_Theta = 0
        return(d_sigma_d_cos_Theta)
    
    alpha = 1/137 #Fine structure constant (SI Units)
    En_MeV = En *1000  #Neutrino/Lepton Energies in MeV
    mn_MeV = mn*1000  #Lepton mass in MeV
    t = (2*En_MeV**2 - mn_MeV**2 - 2*En_MeV*np.sqrt(En_MeV**2 - mn_MeV**2) 
          * cos_Theta) #Transfered momentum^2 (MeV^2)

    leading_terms = (-2* np.sqrt(En_MeV**2 - mn_MeV**2) * Zed**2 * d**2 * alpha)/(En_MeV*t) #MeV^-4
    second_terms = (4*En_MeV**2 - mn_MeV**2 + mn_MeV**4/t) #MeV^2
    Inv_Mev_to_cm = (197.3) * 1e-13 # (MeV^-1 to fm) * (fm to cm)
    d_sigma_d_cos_Theta = Inv_Mev_to_cm**2 * -leading_terms*second_terms #cm^2
    
    return(d_sigma_d_cos_Theta)


def Full_N_d_sigma_d_cos_Theta(d, mn, En, cos_Theta, Zeds, R1s, Ss, num_dens):
    '''
    Determine the differential cross section for a coherent and incoherent 
    scattering at a specified angle with the composition specified
    
    args:
        d: dipole coupling constant in MeV^-1 (float)
        mn: mass of the lepton in GeV (float)
        En: Energy of the lepton in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        Zeds: Atomic numbers (array of ints)
        R1s: Helm effective radii in fm (array of floats, same size as Zeds)
        Ss: Helm skin thicknesses in fm (array of floats, same size as Zeds)
        num_dens: number density of the nucleus in question
                (array of floats, same size as Zeds)
        
    returns:
        N_d_sigma_d_cos_Theta: differential cross section times number dens
                                in cm^-1 (float, same size as En)
    '''
    #Set differential cross section to 0 if the mass is greater than the energy
    if mn >= En:
        d_sigma_d_cos_Theta = 0
        return(d_sigma_d_cos_Theta)
    
    #Calculate the transfered momentum
    En_MeV = En *1000  #Neutrino/Lepton Energies in MeV
    mn_MeV = mn*1000  #Lepton mass in MeV
    t = (2*En_MeV**2 - mn_MeV**2 - 2*En_MeV*np.sqrt(En_MeV**2 - mn_MeV**2) 
          * cos_Theta) #Transfered momentum^2 (MeV^2)
    
    q = np.sqrt(t) #Transfered momentum (MeV)
    
    #Calculate the coherent cross sections for Z = 1
    d_sig_d_cos_coh = d_sigma_d_cos_Theta_coherent(d,mn,En,cos_Theta,1)
    
    #Initialize the cross section as 0
    N_d_sigma_d_cos_Theta = 0
    
    #Iterate through the nuclei
    for Zed_index in range(len(Zeds)):
        Zed = Zeds[Zed_index] #Atomic number
        R1 = R1s[Zed_index] #Effective nuclear radius
        s = Ss[Zed_index]  #Skin thickness
        num_den = num_dens[Zed_index]  #fractional number density of the nucleus
        
        FF2 = formFactorFit.Helm_FF2(q,R1,s) #Form factors^2 for the transferred momentum
        
        N_d_sigma_d_cos_Theta += num_den * d_sig_d_cos_coh * Zed**2 * FF2
    
    return(N_d_sigma_d_cos_Theta)

def N_Cross_Sec_from_radii(d, mn, En, cos_Theta, rs):
    '''
    Determine the differential cross section for a coherent and incoherent 
    scattering at a specified angle with the radius of the interaction specified
    
    args:
        d: dipole coupling constant in MeV^-1 (float)
        mn: mass of the lepton in GeV (float)
        En: Energy of the lepton in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        rs: Normalized radius of the location of the interaction (R_Earth = 1)
            (float, same size as En)
        
    returns:
        N_d_sigma_d_cos_Theta: differential cross section times number dens
                                in cm^-1 (float, same size as En)
        
    '''
    N_d_sigma_d_cos_Theta = np.zeros(len(En))
    for r_index in range(len(rs)):
        r = rs[r_index]
        n_dens = earthComp.n_density(r)
        Zeds,R1s,Ss,num_dens = (np.zeros(len(n_dens)-1),np.zeros(len(n_dens)-1),
                                np.zeros(len(n_dens)-1),np.zeros(len(n_dens)-1))
        
        index = 0
        for element in n_dens.keys():
            if element == 'e':
                continue
            num_dens[index] = n_dens[element] # cm^(-3)
            Zeds[index] = earthComp.atomic_number[element]
            
            #Any element without a Helm fit parameters is treated as Silicon
            try:
                R1s[index] = formFactorFit.Helm_Dict[Element_Dict[element]]["R1"]
                Ss[index] = formFactorFit.Helm_Dict[Element_Dict[element]]["s"]
            except:
                R1s[index] = formFactorFit.Helm_Dict[Element_Dict["Si"]]["R1"]
                Ss[index] = formFactorFit.Helm_Dict[Element_Dict["Si"]]["s"]
            index += 1
        
        N_d_sigma_d_cos_Theta[r_index] = Full_N_d_sigma_d_cos_Theta(d,mn,
                                                                    En[r_index],cos_Theta[r_index],
                                                                    Zeds, R1s, Ss, num_dens)
    
    return(N_d_sigma_d_cos_Theta)

if __name__ == "__main__":
    main()
