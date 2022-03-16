'''
Module to compute properties specific to the mass mixing model.
Calculates decay lengths, branching ratios, and cross sections.

Functions:
    Gamma_tot: Calculates total decay rate
    Gamma_partial: Returns decay rate for a specific channel
    decay_length: Calculates the characteristic decay length
    d_sigma_d_cos_Theta_coherent: Calculates the differential
        cross section for coherently scattering off a nucleus
    d_sigma_d_cos_Theta_nucleon: Calculate the differential
        cross section for scattering off of a nucleon
    n_d_sigma_d_cos_Theta: Calculate the cross section times
        the number density of scattering targets (taking form factors
        into account)
    N_Cross_Sec_from_radii: Calculate the cross section times
        the number density given the radius of the the
        interaction location
'''

def main():

    U=0.1
    mN=0.1
    Enu=100
    cos_Theta=0

    anti_nu="nu"
    sigma_final_nucleon=0
    sigma_final_coherent=0
    for i in range(0,20):
        sigma_final_nucleon+=8*dsigma_dcos_Theta_nucleon(anti_nu,"proton",U,mN,Enu,-1+i/10.0,branch=1)*0.1 \
            + 8*dsigma_dcos_Theta_nucleon(anti_nu,"neutron",U,mN,Enu,-1+i/10.0,branch=1)*0.1
        sigma_final_coherent+=dsigma_dcos_Theta_coherent(U,mN,Enu,-1+i/10.0,8)*0.1

    print("Nucleon cross section:", sigma_final_nucleon)
    print("Coherent cross section:", sigma_final_coherent)
    print("Hello World")
    return 0

#Initialization
import numpy as np #Package for array functions
import formFactorFit
import earthComp
import SampleEvents as sampleEvents
from numpy import random as rand
from numpy import sin, cos

gA=1.27;
MUP=1.79;
MUN=-1.86;
MP=0.94; # Neutron and proton treated as having same mass
MA=1;
MV=0.71;

SW=np.sqrt(0.229);
VUD=0.975;
MPI=139.57E-3;
MPI0=134.9776E-3;
META=547.862E-3;
METAPRIME=957.78E-3;
MK=493.677E-3;
MPHI=1019.461E-3;
MOEMGA=782.65E-3;
MRHO=775.11E-3;
ME=0.510999E-3;
MMU=105.66E-3;
GF=1.1663787E-5;
FPI=130.2E-3;
FK=155.6E-3;
FETA=81.7E-3;
FETAPRIME=-94.7E-3;
GRHO=0.162;
GOMEGA=0.153;
GPHI=0.234;
KAPPARHO=1-SW**2;
KAPPAOMEGA=1.33333*SW**2;
KAPPAPHI=1.3333*SW**2-1;
HBAR_C  =0.1973269804E-18 # GeV km 


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



def Gamma_tot(flav,mN,U):
    '''
    args:
        flav: non-zero flavor mixnig with HNL string options: "e" "mu" or "tau"
          mN: mass of the HNL lepton in GeV (float)
           U:  mixing matrix element. 
        
    returns:
           \Gamma_total in units GeV
    
    actions:
        Computes the decay rate of an HNL summed over all channels below ~ 1 GeV
    '''

    Gamma_List=[ Gamma_3nu(mN,U), \
                 Gamma_2e_nu(flav,mN,U),\
                 Gamma_2mu_nu(flav,mN,U),\
                 Gamma_e_mu_nu(flav,mN,U),\
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
                ]

    return(sum(Gamma_List)) 

def Gamma_partial(flav,mN,U,final_state="nu e e"):
    '''
    args:
           flav:  non-zero flavor mixnig with HNL string options: "e" "mu" or "tau"
             mN:  mass of the HNL lepton in GeV (float)
              U:  mixing matrix element. 
    final_state:  string, default= "nu e e" 
                          options: "nu e e"
                                   "nu mu mu"
                                   "nu e mu"
                                   "nu nu nu"
                                   "pi nu"
                                   "eta nu"
                                   "eta' nu"
                                   "rho nu"
                                   "omega nu"
                                   "phi nu"
                                   "pi e"
                                   "pi mu"
                                   "K e"
                                   "K mu"
                                   "rho e"
                                   "rho mu" 
    returns:
           \Gamma(N-> final state) in units of GeV
    
    actions:
        Computes the decay rate of an HNL summed over all channels below ~ 1 GeV
    '''
    if final_state =="nu e e":
        return(Gamma_2e_nu(flav,mN,U))

    elif final_state =="nu mu mu":
        return(Gamma_2mu_nu(flav,mN,U))

    elif final_state=="nu e mu":
        return(Gamma_e_mu_nu(flav,mN,U))

    elif final_state=="nu nu nu":
        return( Gamma_3nu(mN,U))

    elif final_state=="pi nu":
        return( Gamma_pi_nu(mN,U))

    elif final_state=="eta nu": 
        return( Gamma_eta_nu(mN,U))

    elif final_state=="eta' nu": 
        return( Gamma_eta_prime_nu(mN,U))

    elif final_state=="rho nu": 
        return( Gamma_rho_nu(mN,U))

    elif final_state=="omega nu": 
        return( Gamma_omega_nu(mN,U))

    elif final_state=="phi nu": 
        return( Gamma_phi_nu(mN,U))

    elif final_state=="pi e": 
        return( Gamma_pi_e(flav,mN,U))

    elif final_state=="pi mu": 
        return( Gamma_pi_mu(flav,mN,U))

    elif final_state=="K e": 
        return( Gamma_K_e(flav,mN,U))

    elif final_state=="K mu": 
        return( Gamma_K_mu(flav,mN,U))

    elif final_state=="rho e": 
        return( Gamma_rho_e(flav,mN,U))

    elif final_state=="rho mu": 
        return( Gamma_rho_mu(flav,mN,U))

    else:
        print("Final state not included in code")
        return(0)




def decay_length(flav,U,mN,EN):
    # Returns decay lenght in units of the Earth's radius
    
    R_Earth_in_km = 6378.1
    c_tau=HBAR_C/Gamma_tot(flav,U,mN)/R_Earth_in_km

    if any(EN>mN):
        beta_gamma= EN/mN * np.sqrt(1-mN**2/EN**2)
        Lambda = c_tau*beta_gamma
    else:
        Lambda=0
        
    return(Lambda)



################################################
####
### Needs to be updated
####

def dsigma_dcos_Theta_coherent(U,mN,Enu,cos_Theta,Qw):
    '''
    Determine the differential cross section for a coherent scattering at a specified angle
    
    args:
        U:  mixing matrix element. 
        mn: mass of the lepton in GeV (float)
        En: Energy of the lepton in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        Qw: weak nuclear charge
        
    returns:
        dsigma_dcos_Theta: differential upscattering cross section in cm^2 
                            (float, same size as En)
    
    actions:
        Computes the differential cross section in terms of GeV^{-2}, 
        then converts it to cm^2
    '''
    #Set differential cross section to 0 if the mass is greater than the energy
    if mN >= Enu:
        dsigma_dcos_Theta = 0
        return(dsigma_dcos_Theta)
    

    EN=Enu
    t = (2*EN**2 - mN**2 - 2*EN*np.sqrt(EN**2 - mN**2) * cos_Theta) #Transfered momentum^2 (GeV^2)

    dsigma_dt= U**2*GF**2*Qw**2/(2*np.pi)*(1-0.25*mN**2/Enu**2+0.25*t/Enu**2)
    dAbst_dcos_Theta=2*EN*np.sqrt(EN**2-mN**2)
    Inv_Gev_to_cm = (0.1973) * 1e-13 # (GeV^-1 to fm) * (fm to cm)

    dsigma_dcos_Theta = Inv_Gev_to_cm**2 *(dsigma_dt)*dAbst_dcos_Theta
    
    return(dsigma_dcos_Theta)

                                                
def dsigma_dcos_Theta_nucleon(anti_nu,nucleon,U,mN,Enu,cos_Theta,branch=1):
    '''
    Determine the differential cross section for a coherent scattering at a specified angle
    
    args:
        anti_nu: either "nu" or "nu_bar"
        nucleon: either "proton" or "neutron"
        U:  mixing matrix element. 
        mN: mass of the HNL in GeV (float)
        Enu: Energy of the neutrino in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        branch   : Determines which branch of the EN solution one would like to choose 
                   default = 1 (high energy branch), set to =2 for lower energy branch
        
    returns:
        dsigma_dcos_Theta: differential upscattering cross section in cm^2 
                            (float, same size as En)
    
    actions:
        Computes the differential cross section in terms of GeV^{-2}, then
        converts it to cm^2
    '''
    #Set differential cross section to 0 if the mass is greater than the energy
    if mN >= Enu:
        dsigma_dcos_Theta = 0
        return(dsigma_dcos_Theta)
                                
    if nucleon=="proton":
        tau3=+1
    if nucleon=="neutron":
        tau3=-1

    if anti_nu=="nu":
        sgn=1
    elif anti_nu=="nu_bar":
        sgn=-1
    else:
        print("Options for antinu are 'nu' or 'nu_bar'")

    #Only works with the high energy branch in the case of forward scattering
    EN=sampleEvents.energyOut(Enu,cos_Theta,mN,MP,branch)

    PN=np.sqrt(EN**2-mN**2)
    t=mN**2-2*(Enu*EN -Enu*PN*cos_Theta)
    
    
    error_limit = max(1e-6, abs(t)/1e7)
    assert(np.abs( t+2*MP*(Enu-EN) )<error_limit)

    dAbst_dcos_Theta=2*Enu*PN
    
    s=2*Enu*MP+MP**2
    u=2*MP**2+mN**2-s-t


    eta=-0.25*t/MP**2

    
    
    F2s=0
    F1s=0
    FAs=0
    
    F1=(0.5-SW**2)*tau3*(1+eta*(1+MUP-MUN))/((1+eta)*(1-t/MV**2)**2)\
        -SW**2*(1+eta*(1+MUP+MUN))/((1+eta)*(1-t/MV**2)**2)-F1s/2

    F2=(0.5-SW**2)*tau3*(MUP-MUN)/((1+eta)*(1-t/MV**2)**2) \
        - SW**2*(MUP+MUN)/((1+eta)*(1-t/MV**2)**2) - F2s/2

    FA=0.5*gA*tau3/(1-t/MA**2)**2-FAs/2

    FP=4*MP**2*gA/(MPI**2-t)
    
    A=(mN**2-t)/MP**2 *( (1+eta)*FA**2-(1-eta)*F1**2+eta*(1-eta)*F2**2+4*eta*F1*F2\
                         -0.25*mN**2/MP**2*( (F1+F2)**2+(FA+2*FP)**2-(4-t/MP**2)*FP**2) )
    B= -t/MP**2*FA*(F1+F2)
    C= 0.25*(FA**2+F1**2+eta*F2**2)


    dsigma_dt= U**2*GF**2*MP**2*VUD**2/(8*np.pi*Enu**2)*(A + B*sgn*(s-u)/MP**2  +  C*(s-u)**2/MP**4)

                                                    

    Inv_Gev_to_cm = (0.1973) * 1e-13 # (GeV^-1 to fm) * (fm to cm)
    dsigma_dcos_Theta = Inv_Gev_to_cm**2 *(dsigma_dt)*dAbst_dcos_Theta
    
    return(dsigma_dcos_Theta)

                                            

                                                

def n_dsigma_dcos_Theta(U, mN, Enu, cos_Theta, Zed,  A_minus_Z, R1, S, num_dens,
                               scattering_channel="nucleon",anti_nu="nu",branch=1):
    '''
    Determine the differential cross section for a coherent and incoherent 
    scattering at a specified angle with the composition specified
    
    args:
        U: mixing angle
        mn: mass of the lepton in GeV (float)
        En: Energy of the lepton in GeV (float or array of floats)
        cos_Theta: cosine of the scattering angle (float, same size as En)
        Zeds: Atomic numbers (array of ints)
        R1s: Helm effective radii in fm (array of floats, same size as Zeds)
        Ss: Helm skin thicknesses in fm (array of floats, same size as Zeds)
        num_dens: number density of the nucleus in question
                (array of floats, same size as Zeds)
        Scattering channel: (string) default="nucleon" , 
             options={"nucleon", "coherent","DIS", or "response_function"}
        anti_nu: neutrino vs anti-neutrino (string) options-{"nu","nu_bar"}   
                  does not affect coherent cross section. 
        branch : Branch of the E_N solution function
                 default=1
                 options = 1 (high energy branch) or 2 (low energy branch)
        
    returns:
        N_dsigma_dcos_Theta: differential cross section times number dens
                                in cm^-1 (float, same size as En)
    '''
    #Set differential cross section to 0 if the mass is greater than the energy
    if mN >= Enu:
        dsigma_dcos_Theta = 0
        return(dsigma_dcos_Theta)
    
    #Calculate the transfered momentum
    
    t_coherent = -(2*Enu**2 - mN**2 - 2*Enu*np.sqrt(Enu**2 - mN**2) * cos_Theta) #Transfered momentum^2 (GeV^2)
    
    q_coherent = np.sqrt(-t_coherent)*1000 #Transfered momentum (MeV)
    FF2 = formFactorFit.Helm_FF2(q_coherent,R1,S) #Form factors^2 for the transferred momentum
    if scattering_channel=="coherent":
        
        
        Qw= Zed*(1-4*SW**2) + A_minus_Z  
        
        d_sig_d_cos_coh = dsigma_dcos_Theta_coherent(U,mN,Enu,cos_Theta,Qw)
        N_dsigma_dcos_Theta = num_dens * d_sig_d_cos_coh * FF2 

        
    if scattering_channel=="nucleon":
        #
        # Nucleon level scattering
        #
        # No form factor or anything
        # Could consider coulomb sum rule 
        N_dsigma_dcos_Theta = num_dens*(Zed*dsigma_dcos_Theta_nucleon(anti_nu,"proton",U,mN,Enu,cos_Theta,branch)\
                                      +A_minus_Z*dsigma_dcos_Theta_nucleon(anti_nu,"neutron",U,mN,Enu,cos_Theta,branch))\
            *(1 - FF2)
        
    
    #This is a very simple version of DIS, but it only contributes to the rate
    #   at an order of a few percent
    if scattering_channel == "DIS":
        s = 2 * Enu * MP + MP**2
        if s < mN**2:
            dsigma_dcos_Theta = 0
        else:
            dsigma_dcos_Theta = U**2 * 3E-39 * Enu/2 * np.sqrt(1 - mN**2 / s)
        
        N_dsigma_dcos_Theta = num_dens * dsigma_dcos_Theta * (1-FF2)
    
    if scattering_channel=="response":
        #
        # We can code somethnig up to sample t and omega 
        # given a response function and then calculate 
        # E_nu given that data
        #
        return(0)
    
    #Calculate the coherent cross sections for Z = 1

    return(N_dsigma_dcos_Theta)


def N_Cross_Sec_from_radii(U, mn, Enu, cos_Theta, rs, channel="nucleon" ):
    '''
    Determine the number density weighted differential cross section for  
    scattering at a specified angle with the radius of the interaction specified
    i.e.  \sum_i n_i(r)x d\sigma_i/ d\cos\Theta
    args:
        U: Mass mixing coupling
        mn: mass of the lepton in GeV (float)
        Enu: Energy of neutrino  GeV (array of floats)
        cos_Theta: cosine of the scattering angle (arrray of floats, same size as En)
        rs: Normalized radius of the location of the interaction (R_Earth = 1)
            (array of floats, same size as En)
        channel: Determines scattering channel (string), default="nucleon", options={"coherent", "nucleon"}
        
    returns:
        N_dsigma_dcos_Theta: differential cross section times number dens
                                in cm^-1 (float, same size as En)
        
    '''
    N_dsigma_dcos_Theta = np.zeros(len(Enu))
    for r_index in range(len(rs)):
        r = rs[r_index]
        n_dens = earthComp.n_density(r)
                
        index = 0
        for element in n_dens.keys():
            # Skip over case of electron
            if element == 'e':
                continue
            num_dens = n_dens[element] # cm^(-3)
            A_minus_Z= earthComp.neutron_number[element]
            Zed= earthComp.atomic_number[element]
            
            #Any element without a Helm fit parameters is treated as Silicon
            # We approximate weak form factors by charge form factors
            try:
                R1s = formFactorFit.Helm_Dict[Element_Dict[element]]["R1"]
                Ss = formFactorFit.Helm_Dict[Element_Dict[element]]["s"]
            except:
                R1s = formFactorFit.Helm_Dict[Element_Dict["Si"]]["R1"]
                Ss = formFactorFit.Helm_Dict[Element_Dict["Si"]]["s"]
            
            N_dsigma_dcos_Theta[r_index] += n_dsigma_dcos_Theta(U,mn,Enu[r_index],cos_Theta[r_index],
                                                                         Zed, A_minus_Z, R1s, Ss, num_dens, 
                                                                         scattering_channel = channel)
    
    return(N_dsigma_dcos_Theta)




### Every decay mode
### No doc strings
### Intended for internal use


def lambda_Kallen(a,b,c):
    return(a**2+b**2+c**2-2*a*b-2*a*c-2*b*c)

def Gamma_3nu(mN,U):
    return(4*GF**2*mN**5*U**2/(768*np.pi**3))


def Gamma_e_mu_nu(flav,mN,U):
    x1=ME/mN
    x2=MMU/mN

    # If HNL is heavy enough to make a muon then it is safe
    # at the part per thousand level to neglect electron mass
    # correction are O(me^2/mu^2) \sim (1/200)^2 < 10^{-4}
    
    I_ps=(1-8*x2**2+8*x2**6-x2**8-12*x2**4*np.log(x2**2) )

    if flav=="tau":
        return(0)
    
    if x1+x2>=1:
        return(0)
    else:
        if flav=="e" or flav=="mu":
            return(GF**2*mN**5/(192*np.pi**3)*U**2*I_ps)
        else:
            print("Flavour does not exist")
            return(0)

        
def Gamma_2lep_nu(C1,C2,x,mN,U):
    if x>=0.5:
        return(0)
    else:
        if mN < 0.2:
            L=np.log(1-3*x**2-(1-x**2)*np.sqrt(1-4*x**2))-np.log(x**2*(1+np.sqrt(1-4*x**2)))
        elif mN>= 0.2:
            L = 6 * np.log(2*x) - np.log(x**2*(1+np.sqrt(1-4*x**2)))
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
               *np.sqrt(lambda_Kallen(1,xh**2,xEll*2)/(16*np.pi) ) )
    
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
               *np.sqrt(lambda_Kallen(1,xh**2,xEll**2) ) )


def Gamma_pi_nu(mN,U):
    return(Gamma_P_nu(FPI,MPI0,mN,U) )

def Gamma_eta_nu(mN,U):
    return(Gamma_P_nu(FETA,META,mN,U) )

def Gamma_eta_prime_nu(mN,U):
    return(Gamma_P_nu(FETAPRIME,METAPRIME,mN,U) )



def Gamma_rho_nu(mN,U):
    return(Gamma_V_nu(GRHO,KAPPARHO,MRHO,mN,U) )

def Gamma_omega_nu(mN,U):
    return(Gamma_V_nu(GOMEGA,KAPPAOMEGA,MOEMGA,mN,U) )

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
    assert(flav=="e" or flav=="mu" or flav=="tau")
    
    if flav=="e":
        C1=0.25*(1+4*SW**2+8*SW**4)
        C2=0.5*SW**2*(2*SW**2+1)
    else:
        C1=0.25*(1-4*SW**2+8*SW**4)
        C2=0.5*SW**2*(2*SW**2-1)
        
    
    return(Gamma_2lep_nu(C1,C2,x,mN,U) )

def Gamma_2mu_nu(flav,mN,U):

    x=ME/mN
    assert(flav=="e" or flav=="mu" or flav=="tau")
    
    if flav=="mu":
        C1=0.25*(1+4*SW**2+8*SW**4)
        C2=0.5*SW**2*(2*SW**2+1)
    else:
        C1=0.25*(1-4*SW**2+8*SW**4)
        C2=0.5*SW**2*(2*SW**2-1)
        
    return(Gamma_2lep_nu(C1,C2,x,mN,U) )


## No tau mode because we assume m_N \lesssim  1.5 GeV


if __name__ == "__main__":
    main()
