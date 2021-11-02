def main():
    print("Hello World")
    return 0

#Initialization
import numpy as np #Package for array functions
import formFactorFit
import earthComp
from numpy import random as rand
from numpy import sin, cos


SW=np.sqrt(0.229);
VUD=0.975;
MPI=139.57;
MPI0=134.9776;
META=547.862;
METAPRIME=957.78;
MK=493.677;
MPHI=1019.461;
MOEMGA=782.65;
MRHO=775.11;
ME=0.510999;
MMU=105.66;
GF=1.1663787E-11;
FPI=130.2;
FK=155.6;
FETA=81.7;
FETAPRIME=-94.7;
GRHO=0.162E6;
GOMEGA=0.153E6;
GPHI=0.234E6;
KAPPARHO=1-SW**2;
KAPPAOMEGA=1.33333*SW**2;
KAPPAPHI=1.3333*SW**2-1;
HBAR_C=197.3269804E-18 # MeV km 


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

def lambda_Kallen(a,b,c):
    return(a**2+b**2+c**2-2*a*b-2*a*c-2*b*c)

def Gamma_3nu(mN,U):
    return(4*GF**2*mN**5*U**2/(768*np.pi**3))

def Gamma_2lep_nu(C1,C2,x,mN,U):
    if x>=0.5:
        return(0)
    else:
        L=np.log(1-3*x**2-(1-x**2)*np.sqrt(1-4*x**2))-np.log(x**2*(1+np.sqrt(1-4*x**2)))
        return(GF**2*mN**5*U**2/(192*np.pi**3)*(C1*((1-14*x**2-2*x**4-12*x**6)*np.sqrt(1-4*x**2) + \
                                                12*x**4*(x**4-1)*L)\
                                            + 4*C2*(x**2*(2+10*x**2-12*x**4)*np.sqrt(1-4*x**2) +\
                                                    6*x**4*(1-2*x**2+2*x**4)*L) )\
               )

def Gamma_P_nu(fP,mP,mN,U):
    x=mP/mN
    if x<1:
        return(GF**2*fP*mN**3*U**2*(1-x**2)**2/(32*np.pi) )
    else:
        return(0)

def Gamma_P_ell(fP,mP,mEll,mN,U):
    xh=mP/mN
    xEll=mEll/mN

    if xh+xEll>1:
        return(0)
    else:
        return(GF**2*fP*mN**3*U**2*((1-xEll**2)**2-xh**2*(1+xEll**2))\
               *np.sqrt(lambda_kallen(1,xh**2,xEll*2)/(16*np.pi) ) )

def Gamma_V_nu(gV,kappaV,mV,mN,U):
    xh=mV/mN;

    if xh>=1:
        return(0)
    else:
        return(GF**2*kappaV**2*gV**2*VUD**2*mN**3*U**2/16/np.pi/mV**2*(1-xh**2)**2*(1+2*xh**2) )


def Gamma_V_ell(gV,mV,mEll,mN,U):
    xh=mV/mN
    xEll=mEll/mN

    if xh+xEll>=1:
        return(0)
    else:
        return(GF**2*gV**2*VUD**2*mN**3*U**2/16/np.pi/mV**2*((1-xEll**2)**2+xh**2*(1+xEll**2)-2*xh**4)\
               *np.sqrt(lambda_kallen(1,xh**2,xEll**2) ) )


def Gamma_pi_nu(mN,U):
    return(Gamma_P_nu(FPI,MPI0,mN,U) )

def Gamma_eta_nu(mN,U):
    return(Gamma_P_nu(FETA,META,mN,U) )

def Gamma_eta_prime_nu(mN,U):
    return(Gamma_P_nu(FETAPRIME,METAPRIME,mN,U) )



def Gamma_rho_nu(mN,U):
    return(Gamma_V_nu(GRHO,KAPPARHO,MRHO,mN,U) )

def Gamma_omega_nu(mN,U):
    return(Gamma_V_nu(GOMEGA,KAPPAOMEGA,MOMEGA,mN,U) )

def Gamma_phi_nu(mN,U):
    return(Gamma_V_nu(GPHI,KAPPAPHI,MPHI,mN,U) )


def Gamma_pi_e(flav,mN,U):
    mEll=ME
    if flav=="e":
        return( Gamma_P_ell(FPI,MPI,mEll,mN,U) )
    else:
        return(0)

def Gamma_pi_mu(flav,mN,U):
    mEll=MMU
    if flav=="mu":
        return( Gamma_P_ell(FPI,MPI,mEll,mN,U) )
    else:
        return(0)

def Gamma_K_e(flav,mN,U):
    mEll=ME
    if flav=="e":
        return( Gamma_P_ell(FK,MK,mEll,mN,U) )
    else:
        return(0)

def Gamma_K_mu(flav,mN,U):
    mEll=MMU
    if flav=="mu":
        return( Gamma_P_ell(FK,MK,mEll,mN,U))
    else:
        return(0)

    
## No tau final states because we assume an HNL below 1.5 GeV
## Can revisit if necessary


def Gamma_rho_e(flav,mN,U):
    mEll=ME;
    if flav=="e":
        return(Gamma_V_ell(GRHO,MRHO,mEll,mN,U) )
    else:
        return(0)

def Gamma_rho_mu(flav,mN,U):
    mEll=MMU;
    if flav=="m,u":
        return(Gamma_V_ell(GRHO,MRHO,mEll,mN,U) )
    else:
        return(0)




def Gamma_2e_nu(flav,mN,U):

    x=ME/mN
    assert(flav=="e" or flav=="mu" or flav=="tau"):
    
    if flav=="e":
        C1=0.25*(1+4*SW**2+8*SW**4)
        C2=0.5*SW**2*(2*SW**2+1)
    else:
        C1=0.25*(1-4*SW**2+8*SW**4)
        C2=0.5*SW**2*(2*SW**2-1)
        
    
    return(Gamma_2lep_nu(C1,C2,x,mN,U) )

def Gamma_2mu_nu(flav,mN,U):

    x=ME/mN
    assert(flav=="e" or flav=="mu" or flav=="tau"):
    
    if flav=="mu":
        C1=0.25*(1+4*SW**2+8*SW**4)
        C2=0.5*SW**2*(2*SW**2+1)
    else:
        C1=0.25*(1-4*SW**2+8*SW**4)
        C2=0.5*SW**2*(2*SW**2-1)
        
    return(Gamma_2lep_nu(C1,C2,x,mN,U) )


## No tau mode because we assume m_N< 1.5 GeV or so 


def Gamma_tot(flav,mN,U):

    Gamma_List=[ Gamma_3nu(mN,U), \
                 Gamma_pi_nu(mN,U),\
                 Gamma_eta_nu(mN,U),\
                 Gamma_eta_prime_nu(mN,U),\
                 Gamma_rho_nu(mN,U),\
                 Gamma_omega_nu(mN,U),\ 
                 Gamma_phi_nu(mN,U),\ 
                 Gamma_pi_e(flav,mN,U),\
                 Gamma_pi_mu(flav,mN,U),\
                 Gamma_K_e(flav,mN,U),\
                 Gamma_K_mu(flav,mN,U),\
                 Gamma_rho_e(flav,mN,U),\
                 Gamma_rho_mu(flav,mN,U),\
                 Gamma_2e_nu(flav,mN,U),\
                 Gamma_2mu_nu(flav,mN,U)\
                ]

    return(sum(Gamma_List)) 


def decay_length(flav,U,mN,EN):
    # Returns decay lenght in units of the Earth's radius
    
    R_Earth_in_km = 6378.1
    c_tau=HBAR_C*Gamma_total(flav,U,mN)/R_Earth_in_km

    if EN>mN:
        beta_gamma= EN/mN * np.sqrt(1-mN**2/EN**2)
        Lambda = c_tau*beta_gamma
    else:
        Lambda=0
        
    return(Lambda)



################################################
####
####  End of edits
####
####

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
