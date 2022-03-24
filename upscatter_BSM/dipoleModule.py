def main():
    print("Hello World")
    return 

'''
This module is specific for the HNL Dipole coupling portal.  It calculates the decay length
    of the HNL and the scattering cross sections.
Functions:
    decay_length: Calculate the characteristic decay length of the HNL
    d_sigma_d_cos_Theta_coherent: Calculate the differential cross section for
        a neutrino scattering off of a nucleus coherently.
    Calc_F1_F2: Calcuate the form factors of for protons and neutrons at a given
        value of Q.
    HNL_Energy: Determine the energy of the HNL after scattering
    dEN_d_cos_Theta: Calculate derivative of HNL energy with respect to scattering angle
    d_t_d_cos_Theta: Derivative of Mandelstam variable t with respect to scattering angle
    d_sigma_d_t_incoh: Differential cross section with respect to Mandelstam variable t
    d_sigma_d_cos_Theta: Differential cross section with respect to cosine
        of scattering angle
    Full_N_d_sigma_d_cos_Theta: Calculate the full differential cross section 
        (coherent and incoherent) times the number density of nuclei for a given list 
        of nuclei
    N_Cross_Sec_from_radii: Calculate the full differential cross section times the 
        number density of nuclei given the radius (radius determines the types of nuclei
        and their number density).
'''

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
    
    R_Earth = 1
    mn_MeV = mn*1000  #Convert mass to MeV
    En_MeV = En*1000  #Convert energy to MeV
    try:
        Lambda = np.zeros(len(En))
        good_indices = En >= mn
        Lambda[good_indices] = (R_Earth * (1.97e-9/d)**2 *(1/mn_MeV)**4 \
                                * (En_MeV[good_indices]/10) * \
                                    np.sqrt((1-mn**2/En[good_indices]**2)/0.99) )#Decay lengths (cm)
    
    except:
        if mn > En:
            Lambda = 0
        else:
            Lambda= (R_Earth * (1.97e-9/d)**2 *(1/mn_MeV)**4 \
                                * (En_MeV/10) * \
                                    np.sqrt((1-mn**2/En**2)/0.99) )#Decay lengths (cm)
    
    #Set decay length to 0 if mass is greater than energy
    #Lambda = Lambda * np.heaviside(En - mn,0)
    
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
    En_MeV = En *1000  #Neutrino/HNL Energies in MeV
    mn_MeV = mn*1000  #HNL mass in MeV
    t = (2*En_MeV**2 - mn_MeV**2 - 2*En_MeV*np.sqrt(En_MeV**2 - mn_MeV**2) 
          * cos_Theta) #Transfered momentum^2 (MeV^2)

    leading_terms = (-2* np.sqrt(En_MeV**2 - mn_MeV**2) * Zed**2 * d**2 * alpha)/(t * En_MeV) #MeV^-4
    second_terms = (4*En_MeV**2 - mn_MeV**2 + mn_MeV**4/t) #MeV^2
    Inv_Mev_to_cm = (197.3) * 1e-13 # (MeV^-1 to fm) * (fm to cm)
    d_sigma_d_cos_Theta = Inv_Mev_to_cm**2 * -leading_terms*second_terms #cm^2
    
    
    return(d_sigma_d_cos_Theta)


def Calc_F1_F2(Q):
    '''
    Calculates the Form Factors for protons and neutrons
    
    args:
        Q: Momentum Transfer in GeV (float or array)
    returns:
        F1p: Form factor value for proton
        F2p: Form factor value for proton
        F1n: Form factor value for neutron
        F2n: Form factor value for neutron
    '''
    m_nuc = 0.94 #Nucleon mass GeV
    mu_p = 2.793
    mu_n = -1.913
    G_D = (1 + Q**2 /0.71)**(-2)
    
    G_E_p = G_D
    G_E_n = 0
    G_M_p = mu_p * G_D
    G_M_n = mu_n * G_D
    
    F1p = 1/(1 + Q**2 / (4 * m_nuc**2)) * (G_E_p + Q**2 /(4 * m_nuc**2) * G_M_p)
    F1n = 1/(1 + Q**2 / (4 * m_nuc**2)) * (G_E_n + Q**2 /(4 * m_nuc**2) * G_M_n)
    F2p = 1/(1 + Q**2 / (4 * m_nuc**2)) * (G_M_p - G_E_p)
    F2n = 1/(1 + Q**2 / (4 * m_nuc**2)) * (G_M_n - G_E_n)
    
    return(F1p,F2p,F1n,F2n)

def HNL_Energy(E_nu, m_A, m_N, cos_Theta):
    '''
    Function that returns the energy of the HNL, assuming the target
        is at rest
    args:
        E_nu: Energy of incoming neutrino in GeV (float, array)
        m_A: Mass of the scattering target in GeV (float)
        m_N: Mass of the HNL in GeV (float)
        cos_Theta: Cosine of scattering angle (float, array same size as E_nu)
        
    returns:
        E_N: Energy of the HNL in GeV (float, array)
    '''
    
    first_num = (E_nu+m_A)*(m_N**2 + 2 *E_nu* m_A)
    first_den = 2 * (E_nu**2 * (1-cos_Theta**2) + 2 * E_nu * m_A + m_A**2)
    
    discrim = -4 * m_N**2 *(m_A**2 + E_nu**2 * (1-cos_Theta**2)) + (m_N**2 - 2* E_nu * m_A)**2
    sec_den = 2 *(E_nu**2 * (1-cos_Theta**2)+ 2 * E_nu* m_A + m_A**2)
    
    E_N = (first_num/first_den) + E_nu * cos_Theta* np.sqrt(discrim)/sec_den
    
    return(E_N)

def d_EN_d_cos_Theta(E_nu, m_A, m_N, cos_Theta):
    '''
    Derivative of HNL Energy with respect to cos(Theta)
    args:
        E_nu: Energy of incoming neutrino in GeV (float, array)
        m_A: Mass of the scattering target in GeV (float)
        m_N: Mass of the HNL in GeV (float)
        cos_Theta: Cosine of scattering angle (float, array same size as E_nu)
        
    returns:
        deriv: derivative of HNL Energy wrt cos(Theta) in GeV
    '''
    discrim = (m_N**2 - 2* E_nu * m_A)**2 - 4* m_N**2 * (E_nu**2 *(1 - cos_Theta**2) + m_A**2)
    common_den = E_nu**2 * (1 - cos_Theta**2) + 2 * E_nu * m_A + m_A**2
    
    first = E_nu**3 * cos_Theta**2 *np.sqrt(discrim)/(common_den**2)
    second = 2 * E_nu**3 * m_N**2 * cos_Theta**2/ (common_den * np.sqrt(discrim))
    third = E_nu**2 * cos_Theta * (E_nu + m_A) * (2 * E_nu * m_A + m_N**2)/ common_den**2
    fourth = E_nu * np.sqrt(discrim)/(2*common_den)
    
    deriv = first + second + third + fourth
    return(deriv)

def d_t_d_cos_Theta(E_nu, m_A, m_N, cos_Theta):
    '''
    Derivative of Mandelstam variable t with respect to cos(Theta)
    args:
        E_nu: Energy of incoming neutrino in GeV (float, array)
        m_A: Mass of the scattering target in GeV (float)
        m_N: Mass of the HNL in GeV (float)
        cos_Theta: Cosine of scattering angle (float, array same size as E_nu)
        
    returns:
        deriv: derivative of t in GeV^2 wrt cos(Theta)
    '''
    E_N = HNL_Energy(E_nu,m_A,m_N,cos_Theta)
    d_E_d_cos = d_EN_d_cos_Theta(E_nu,m_A,m_N,cos_Theta)
    p_N = np.sqrt(E_N**2 - m_N**2)
    deriv = -2*E_nu * (d_E_d_cos - p_N - (E_N/p_N) *d_E_d_cos * cos_Theta) #GeV^2
    return(deriv)

def d_sigma_d_t_incoh(E_nu,t, m_N,d,F1,F2):
    '''
    Calculate the differential incoherent cross section wrt t
    args:
        E_nu: Energy of incoming neutrino in GeV(float,array)
        t: Mandelstam variable GeV^2 (float, array)
        m_N: mass of the HNL in GeV (float)
        d: dipole coupling strength (MeV^-1)
        F1: Constant for nucleon
        F2: Constant for nucleon
    returns:
        d_s_d_t: differential cross section

    '''
    
    m_p = 0.94 # Nucleon mass (GeV)
    fine_struc = 1/137
    first = m_N**2 * t - 2* m_N**4 + t**2 #GeV
    second = ((2*F1**2 *m_p**2 *(2*m_p**2 + t)) - (12*F1*F2*m_p**2 * t)
              + (F2**2 * t * (8*m_p**2 + t))) #GeV^4
    den = m_p**2 * t**2 * (2* E_nu * m_p)**2 #GeV^-8
    
    d_s_d_t = fine_struc * d**2 * first * second/den #GeV^-2 MeV^-2
    return(d_s_d_t)

def d_sigma_d_cos_Theta_incoh(E_nu,cos_Theta, m_N,d,F1,F2):
    '''
    Calculate the differential incoherent cross section wrt cos(Theta)
    args:
        E_nu: Energy of incoming neutrino in GeV(float,array)
        cos_Theta: cosine of scattering angle (float,array)
        m_N: mass of the HNL in GeV (float)
        d: dipole coupling strength (MeV^-1)
        F1: Constant for nucleon
        F2: Constant for nucleon
    returns:
        d_s_d_cos: differential cross section (cm^2)

    '''
    m_p = 0.94 # Nucleon mass (GeV)
    E_N = HNL_Energy(E_nu,m_p,m_N,cos_Theta) #GeV
    
    t = m_N**2 - 2*E_nu*(E_N - np.sqrt(E_N**2 - m_N**2)*cos_Theta) #GeV^2
    
    d_s_d_t = d_sigma_d_t_incoh(E_nu,t,m_N,d,F1,F2) #GeV^-2 MeV^-2
    d_t_d_cos = d_t_d_cos_Theta(E_nu,m_p,m_N,cos_Theta) #GeV^2
    
    d_s_d_cos = d_s_d_t * d_t_d_cos #MeV^-2
    Inv_Mev_to_cm = (197.3) * 1e-13 # (MeV^-1 to fm) * (fm to cm)
    d_s_d_cos = abs(d_s_d_cos* (Inv_Mev_to_cm)**2 )#cm^2
    return(d_s_d_cos)

def Full_N_d_sigma_d_cos_Theta(d, mn, E_nu, cos_Theta,
                               Zeds,As, R1s, Ss, num_dens):
    '''
    Determine the differential cross section for a coherent and incoherent 
    scattering at a specified angle with the composition specified
    
    args:
        d: dipole coupling constant in MeV^-1 (float)
        mn: mass of the lepton in GeV (float)
        E_nu: Energy of the incoming neutrino in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        Zeds: Atomic numbers (array of ints)
        As: Atomic masses (array of floats)
        R1s: Helm effective radii in fm (array of floats, same size as Zeds)
        Ss: Helm skin thicknesses in fm (array of floats, same size as Zeds)
        num_dens: number density of the nucleus in question
                (array of floats, same size as Zeds)
        
    returns:
        N_d_sigma_d_cos_Theta: differential cross section times number dens
                                in cm^-1 (float, same size as En)
    '''
    #Set differential cross section to 0 if the mass is greater than the energy
    if mn >= E_nu:
        d_sigma_d_cos_Theta = 0
        return(d_sigma_d_cos_Theta)
    
    m_nuc = 0.94 #nucelon mass GeV
    
    #Calculate the transfered momentum
    E_N_coh_MeV = E_nu * 1000 #HNL Energy in MeV
    E_nu_MeV = E_nu *1000  #Neutrino Energies in MeV
    mn_MeV = mn*1000  #HNL mass in MeV
    t_coh_MeV = (2*E_nu_MeV*E_N_coh_MeV - mn_MeV**2 - 2*E_nu_MeV*np.sqrt(E_N_coh_MeV**2 - mn_MeV**2) 
          * cos_Theta) #Transfered momentum^2 (MeV^2)
    
    q_coh_MeV = np.sqrt(t_coh_MeV) #Transfered momentum (MeV)
    
    E_N_incoh = HNL_Energy(E_nu,m_nuc,mn,cos_Theta) #Incoherent HNL Energy GeV
    t_incoh = mn**2 - 2*E_nu * (E_N_incoh - np.sqrt(E_N_incoh**2 - mn**2)*cos_Theta) #GeV^2
    q_incoh = np.sqrt(-t_incoh) #GeV
    
    (F1p,F2p,F1n,F2n) = Calc_F1_F2(q_incoh)
    
    #Calculate the coherent cross sections for Z = 1
    d_sig_d_cos_coh = d_sigma_d_cos_Theta_coherent(d,mn,E_nu,cos_Theta,1)
    
    #Calculate the incoherent cross sections for protons and neutrons
    d_sig_d_cos_pro = d_sigma_d_cos_Theta_incoh(E_nu,cos_Theta,mn,d,F1p,F2p)
    d_sig_d_cos_neut = d_sigma_d_cos_Theta_incoh(E_nu,cos_Theta,mn,d,F1n,F2n)
    '''
    if d_sig_d_cos_coh < 0:
        print('coh problem')
    if d_sig_d_cos_pro < 0:
        print('pro problem')
    if d_sig_d_cos_neut < 0:
        print('neut prob')
    '''

    #Initialize the cross section as 0
    N_d_sigma_d_cos_Theta = 0
    
    #Iterate through the nuclei
    for Zed_index in range(len(Zeds)):
        Zed = Zeds[Zed_index] #Atomic number
        A = As[Zed_index]
        R1 = R1s[Zed_index] #Effective nuclear radius
        s = Ss[Zed_index]  #Skin thickness
        num_den = num_dens[Zed_index]  #fractional number density of the nucleus
        
        FF2 = formFactorFit.Helm_FF2(q_coh_MeV,R1,s) #Nucleus Form factors^2 for the transferred momentum
        
        N_d_sigma_d_cos_Theta += num_den * (d_sig_d_cos_coh * Zed**2 * FF2
                                            + d_sig_d_cos_pro * Zed * (1-FF2)
                                            + d_sig_d_cos_neut * (A-Zed) )
        
    #Remove cases where mn > E_nu or E_HNL is not greater than 0    
    try:
        bad_indices1 = mn > E_nu
        N_d_sigma_d_cos_Theta[bad_indices1] = np.zeros(sum(bad_indices1))
        
    except:
        ignore = 0
    
    try:
        for E_HNL_index in range(len(E_N_incoh)):
            if not E_N_incoh[E_HNL_index] > 0:
                N_d_sigma_d_cos_Theta[E_HNL_index] = 0
    except:
        if not E_N_incoh > 0:
            N_d_sigma_d_cos_Theta = 0
                
    
    return(N_d_sigma_d_cos_Theta)

def N_Cross_Sec_from_radii(d, mn, E_nu, cos_Theta, rs):
    '''
    Determine the differential cross section for a coherent and incoherent 
    scattering at a specified angle with the radius of the interaction specified
    
    args:
        d: dipole coupling constant in MeV^-1 (float)
        mn: mass of the lepton in GeV (float)
        E_nu: Energy of incoming neutrino in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        rs: Normalized radius of the location of the interaction (R_Earth = 1)
            (float, same size as En)
        
    returns:
        N_d_sigma_d_cos_Theta: differential cross section times number dens
                                in cm^-1 (float, same size as En)
        
    '''
    try:
        N_d_sigma_d_cos_Theta = np.zeros(len(E_nu))
        for r_index in range(len(rs)):
            r = rs[r_index]
            n_dens = earthComp.n_density(r)
            Zeds,As,R1s,Ss,num_dens = (np.zeros(len(n_dens)-1),np.zeros(len(n_dens)-1),
                                    np.zeros(len(n_dens)-1),np.zeros(len(n_dens)-1),
                                    np.zeros(len(n_dens)-1))
            
            index = 0
            for element in n_dens.keys():
                if element == 'e':
                    continue
                num_dens[index] = n_dens[element] # cm^(-3)
                Zeds[index] = earthComp.atomic_number[element]
                As[index] = earthComp.molar_mass[element]
                
                #Any element without a Helm fit parameters is treated as Silicon
                try:
                    R1s[index] = formFactorFit.Helm_Dict[Element_Dict[element]]["R1"]
                    Ss[index] = formFactorFit.Helm_Dict[Element_Dict[element]]["s"]
                except:
                    R1s[index] = formFactorFit.Helm_Dict[Element_Dict["Si"]]["R1"]
                    Ss[index] = formFactorFit.Helm_Dict[Element_Dict["Si"]]["s"]
                index += 1
            
            N_d_sigma_d_cos_Theta[r_index] = Full_N_d_sigma_d_cos_Theta(d,mn,
                                                                        E_nu[r_index],cos_Theta[r_index],
                                                                        Zeds, As, R1s, Ss, num_dens)
    except:
        r = rs
        n_dens = earthComp.n_density(r)
        Zeds,As,R1s,Ss,num_dens = (np.zeros(len(n_dens)-1),np.zeros(len(n_dens)-1),
                                    np.zeros(len(n_dens)-1),np.zeros(len(n_dens)-1),
                                    np.zeros(len(n_dens)-1))
            
        index = 0
        for element in n_dens.keys():
            if element == 'e':
                continue
            num_dens[index] = n_dens[element] # cm^(-3)
            Zeds[index] = earthComp.atomic_number[element]
            As[index] = earthComp.molar_mass[element]
            
            #Any element without a Helm fit parameters is treated as Silicon
            try:
                R1s[index] = formFactorFit.Helm_Dict[Element_Dict[element]]["R1"]
                Ss[index] = formFactorFit.Helm_Dict[Element_Dict[element]]["s"]
            except:
                R1s[index] = formFactorFit.Helm_Dict[Element_Dict["Si"]]["R1"]
                Ss[index] = formFactorFit.Helm_Dict[Element_Dict["Si"]]["s"]
            index += 1
        
        N_d_sigma_d_cos_Theta = Full_N_d_sigma_d_cos_Theta(d,mn,
                                                           E_nu,cos_Theta,
                                                           Zeds, As, R1s, Ss, num_dens)
        
    return(N_d_sigma_d_cos_Theta)
    

if __name__ == "__main__":
    main()
