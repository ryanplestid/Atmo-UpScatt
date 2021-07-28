def main():

    x=np.linspace(0,6000,6000) # km 
    ne=2+np.sin(x)   # n_avagadro/cm**3

    E_nu=0.1 # GeV
    
    print(   getProbs(x,ne,E_nu)  )
    


import numpy as np
from  scipy import interpolate
from scipy.integrate import complex_ode

from scipy.integrate import solve_ivp
from numpy import cos,sin,exp,log,sqrt,pi


###
### Assumes normal hierarchy
### 
###
m21sq=7.42*1E-5  #eV^2
m31sq=2.514*1E-3 #eV^2
theta12=33.34*(2*pi/360)
theta13=8.57*(2*pi/360)
theta23=49.2*(2*pi/360)
delta=195*(2*pi/360)
G_F=1.116*1E-5  #GeV^{-2}

rot1=np.array( [ [cos(theta12) ,sin(theta12),  0],
              [-sin(theta12),cos(theta12),  0],
              [       0     ,     0      ,  1]
            ])

rot2=np.array( [[cos(theta13) ,0,  sin(theta13)*exp(-1j*delta)],
                [0,           1 ,              0              ],
                [-sin(theta13)*exp(1j*delta) ,0, cos(theta13)]
                ]
              )


rot3=np.array( [[1    ,     0     ,       0      ],
                [0,  cos(theta23) , sin(theta23) ],
                [0, -sin(theta23) , cos(theta23) ]
                ]
              )
PMNS=rot3@rot2@rot1

mass_diag=np.diag([0,m21sq,m31sq])
mass_flav=PMNS@mass_diag@(PMNS.conj().T)





def getProbs(x,ne,E_nu,anti=False):
    '''
    Computes dictionary of transiion probabilities for neutrinos propagating
    along a trajectory with number density ne(x) 

    args: 

         x: Array of positions in unts of km 
         
         ne: Array of electron number densities in units of Avagadro's number/cm^3

         E_nu: Energy of neutino in GeV

         anti: Boolean variable that indicates if neutrino is particle or anti-partcle

    returns:

           Dictinoary with P_ee etc. Format is { "e->e" : # , ... } 
    '''
    xMax=x[-1:]


    assert np.amin(ne) >= 0, "Negative number density not allowed "

    
    get_ne=interpolate.interp1d(x,ne)

    def matt_pot(x):
        return(sqrt(8)*G_F*E_nu*np.diag([get_ne(x),0,0]) )


    # Should have units of 1/km
    #
    # For vacuum term:  eV^2/GeV= 1E-9 eV = 5.07 km^-1
    #
    # For matter term:  GeV^{-2} cm^{-3} * avagadro's number = 23.7 km^-1 
    def ham(x, psi):
        if anti:
            return -1j*(5.07*mass_flav - 23.7*matt_pot(x) )@psi/(2*E_nu)
        else:
            return -1j*(5.07*mass_flav + 23.7*matt_pot(x) )@psi/(2*E_nu)

    probs={}


    ### Electron neutrino initial state
    sol = solve_ivp(ham, [0,xMax], [1+0*1j, 0+0*1j, 0])
    final=np.transpose(sol.y)[-1:][0]
    
    err=np.abs(np.vdot(final,final))-1

    if  err>1E-2:
        sol = solve_ivp(ham, [0,xMax], [0+0*1j, 1+0*1j, 0],rtol=1E-6)
        final=np.transpose(sol.y)[-1:][0]
    
        err=np.abs(np.vdot(final,final))-1

    assert err  < 1E-2,  "Matter propagation not accurate enough. Error is {}".format(err) 


    probs["e->e"]=np.abs(final[0])**2
    probs["e->mu"]=np.abs(final[1])**2
    probs["e->tau"]=np.abs(final[2])**2


    ### Muon  neutrino initial state
    sol = solve_ivp(ham, [0,xMax], [0+0*1j, 1+0*1j, 0])
    final=np.transpose(sol.y)[-1:][0]
    
    err=np.abs(np.vdot(final,final))-1

    if  err>1E-2:
        sol = solve_ivp(ham, [0,xMax], [0+0*1j, 1+0*1j, 0],rtol=1E-6)
        final=np.transpose(sol.y)[-1:][0]
    
        err=np.abs(np.vdot(final,final))-1
    
    assert err  < 1E-2,  "Matter propagation not accurate enough. Error is {}".format(err) 


    probs["mu->e"]=np.abs(final[0])**2
    probs["mu->mu"]=np.abs(final[1])**2
    probs["mu->tau"]=np.abs(final[2])**2


    ### Tau neutrino initial state
    sol = solve_ivp(ham, [0,xMax], [0+0*1j, 0+0*1j, 1])
    final=np.transpose(sol.y)[-1:][0]

    err=np.abs(np.vdot(final,final))-1


    if  err>1E-2:
        sol = solve_ivp(ham, [0,xMax], [0+0*1j, 1+0*1j, 0],rtol=1E-6)
        final=np.transpose(sol.y)[-1:][0]
    
        err=np.abs(np.vdot(final,final))-1

    assert err  < 1E-2,  "Matter propagation not accurate enough. Error is {}".format(err) 

    probs["tau->e"]=np.abs(final[0])**2
    probs["tau->mu"]=np.abs(final[1])**2
    probs["tau->tau"]=np.abs(final[2])**2

    return(probs)
    



if __name__ == '__main__':
    main()
