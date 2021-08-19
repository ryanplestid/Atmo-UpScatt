def main():
    print("Hello World")
    return 0

#Initialization
import numpy as np #Package for array functions
import formFactorFit
import earthComp
import dipoleModel
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
        d_s_d_cos: differential cross section

    '''
    m_p = 0.94 # Nucleon mass (GeV)
    E_N = HNL_Energy(E_nu,m_p,m_N,cos_Theta) #GeV
    
    t = m_N**2 - 2*E_nu*(E_N - np.sqrt(E_N**2 - m_N**2)*cos_Theta) #GeV^2
    
    d_s_d_t = d_sigma_d_t_incoh(E_nu,t,m_N,d,F1,F2) #GeV^-2 MeV^-2
    d_t_d_cos = d_t_d_cos_Theta(E_nu,m_p,m_N,cos_Theta) #GeV^2
    
    d_s_d_cos = d_s_d_t * d_t_d_cos #MeV^-2
    Inv_Mev_to_cm = (197.3) * 1e-13 # (MeV^-1 to fm) * (fm to cm)
    d_s_d_cos = d_s_d_cos* (Inv_Mev_to_cm)**2 #cm^2
    return(-d_s_d_cos)

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
    mn_MeV = mn*1000  #Lepton mass in MeV
    t_coh_MeV = (2*E_nu_MeV*E_N_coh_MeV - mn_MeV**2 - 2*E_nu_MeV*np.sqrt(E_N_coh_MeV**2 - mn_MeV**2) 
          * cos_Theta) #Transfered momentum^2 (MeV^2)
    
    q_coh_MeV = np.sqrt(t_coh_MeV) #Transfered momentum (MeV)
    
    E_N_incoh = HNL_Energy(E_nu,m_nuc,mn,cos_Theta) #Incoherent HNL Energy GeV
    t_incoh = mn**2 - 2*E_nu * (E_N_incoh - np.sqrt(E_N_incoh**2 - mn**2)*cos_Theta) #GeV^2
    q_incoh = np.sqrt(-t_incoh) #GeV
    
    (F1p,F2p,F1n,F2n) = Calc_F1_F2(q_incoh)
    
    #Calculate the coherent cross sections for Z = 1
    d_sig_d_cos_coh = dipoleModel.d_sigma_d_cos_Theta_coherent(d,mn,E_nu,cos_Theta,1)
    
    #Calculate the incoherent cross sections for Z = 1
    d_sig_d_cos_pro = d_sigma_d_cos_Theta_incoh(E_nu,cos_Theta,mn,d,F1p,F2p)
    d_sig_d_cos_neut = d_sigma_d_cos_Theta_incoh(E_nu,cos_Theta,mn,d,F1n,F2n)
    
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
    
    return(N_d_sigma_d_cos_Theta)
    

if __name__ == "__main__":
    main()