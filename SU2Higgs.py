import numpy as np
import matplotlib.pyplot as pl
from numba import jit

#Parameters
L = 16
N_time = 2
self_coupling = 0.5 
kappas = np.arange(0,5,0.2)
beta = 8
N_dimensions = 2
number_of_measurements = 100
number_of_updates = 2000
thermalization = 50
C=0.5

#Initialize the measurements
avg_scalars = np.zeros([len(kappas),number_of_measurements], dtype=float)
mean_scalars = np.zeros(len(kappas), dtype=float)
sigma_scalars = np.zeros(len(kappas), dtype=float)

#Define the generators of the SU(2) group, which
#are the Pauli matrices
generators = np.zeros([3,2,2], dtype = complex)
generators[0] = np.array([[0,1],[1,0]])
generators[1] = np.array([[0,-1j],[1j,0]])
generators[2] = np.array([[1,0],[0,-1]])


#==============================================================================
#Auxiliary functions for the simulation:

#Move one unit up to direction d
@jit
def coord_up(t,x,d):
    if d==0:
        return (t+1)%N_time, x
    if d==1: 
        return t, (x+1)%L

#Move one unit down to direction d
@jit
def coord_dn(t,x,d):
    if d==0:
        return t-1, x
    if d==1: 
        return t, x-1

#Calculate a plaguette
@jit
def plaguette(t,x,d1,d2,links):
    t1, x1 = coord_up(t,x,d1)
    t2, x2 = coord_up(t,x,d2)
    return np.dot(links[t][x][d1], np.dot(links[t1][x1][d2],
            np.dot(links[t2][x2][d1].conj().T, links[t][x][d2].conj().T)))

#Calculate the term where a link is squeezed between 
#two scalar fields at different sites
@jit
def coupling(t,x,d,scalar,links):
    t1,x1 = coord_up(t,x,d)
    return  np.dot( scalar[t][x].conj(), np.dot( links[t][x][d], 
            scalar[t1][x1]))

#Calculate the part of the action which contributes to the
#Metropolis probability
@jit
def local_action(t,x,d,scalar,links,kappa):
    action = 0
    action += np.dot(scalar[t][x].conj(),scalar[t][x])
    action += self_coupling*( np.dot(scalar[t][x].conj(),scalar[t][x]) - 1)**2
    for d1 in range(N_dimensions):
        t1, x1 = coord_dn(t,x,d1)
        action -= 2*kappa*np.real( coupling(t,x,d1,scalar,links)
                + coupling(t1,x1,d1,scalar,links) )
        if d1 != d:
            action -= 0.5 * beta *np.real(np.trace(plaguette(t,x,d,d1,links) + 
                    plaguette(t1,x1,d,d1,links)))
    return action

#Update the lattice number_of_updates times. The matrices
# and scalar field are split to components so that Numba does
# not print any warnings. It seems that Numba does not like when
# matrices are allocated inside the function.
@jit
def update(scalar,links,kappa,generators):

    for j in range(number_of_updates):
        
        #Pick random point and random direction from the lattice
        t = int( N_time*np.random.random() )
        x = int( L*np.random.random() )
        d = int( N_dimensions*np.random.random() )     

        #Save the old link and scalar field and calculate the 
        # action
        old_link1 = links[t][x][d][0][0]
        old_link2 = links[t][x][d][0][1]
        old_link3 = links[t][x][d][1][0]
        old_link4 = links[t][x][d][1][1]

        old_scalar1 = scalar[t][x][0] 
        old_scalar2 = scalar[t][x][1] 
        
        action_now = local_action(t,x,d,scalar,links,kappa)

        #Update the link and scalar field and calculate the action    
        i = int( 3*np.random.random() )

        delta = np.random.normal()
        links[t][x][d][0][0] = np.dot(np.cos(C*delta)*np.identity(2) 
                + 1j*np.sin(C*delta)*generators[i],links[t][x][d])[0][0]
        links[t][x][d][0][1] = np.dot(np.cos(C*delta)*np.identity(2) 
                + 1j*np.sin(C*delta)*generators[i],links[t][x][d])[0][1]
        links[t][x][d][1][0] = np.dot(np.cos(C*delta)*np.identity(2) 
                + 1j*np.sin(C*delta)*generators[i],links[t][x][d])[1][0]
        links[t][x][d][1][1] = np.dot(np.cos(C*delta)*np.identity(2) 
                + 1j*np.sin(C*delta)*generators[i],links[t][x][d])[1][1]

        scalar[t][x][0] = (np.real(scalar[t][x][0]) + C*np.random.normal()
                + np.imag(scalar[t][x][0]) + C*np.random.normal())
        scalar[t][x][1] = (np.real(scalar[t][x][1]) + C*np.random.normal()
                + np.imag(scalar[t][x][1]) + C*np.random.normal())
        
        action_new = local_action(t,x,d,scalar,links,kappa)

        #Calculate the Metropolis probability and either
        # keep the update or go back to the initial values
        diff = np.real(action_new - action_now)
        var = np.exp(-diff)
        if var >= 1:
            probability = 1
        else:
            probability = var

        if np.random.random() >= probability:
            links[t][x][d][0][0] = old_link1
            links[t][x][d][0][1] = old_link2
            links[t][x][d][1][0] = old_link3
            links[t][x][d][1][1] = old_link4

            scalar[t][x][0] = old_scalar1
            scalar[t][x][1] = old_scalar2 
    return

#==============================================================================
#The actual simulation:

#Loop through the different kappas
for k in range(len(kappas)): 
   
    #The lattice needs to be initialized for every kappa
    scalar = np.zeros([N_time,L,2], dtype=complex)
    links = np.zeros([N_time,L,N_dimensions,2,2], dtype=complex)
    links[:][:][:] = np.array([[1,0],[0,1]])
    
    #Thermalization
    for i in range(thermalization):

        update(scalar,links,kappas[k],generators)
        
    #Start measuring    
    for i in range(number_of_measurements):
        
        update(scalar,links,kappas[k],generators)
        
        #Perform the measurement
        avg_scalar = 0
        for t in range(N_time):
            for x in range(L):
                avg_scalar += np.dot(np.conj(scalar[t][x]),scalar[t][x])

        avg_scalars[k,i] = 1/(N_time * L) * np.real(avg_scalar) 

#Calculate the expectation values
for k in range(len(kappas)): 
    mean_scalars[k] = np.mean(avg_scalars[k,:])
    sigma_scalars[k] = np.std(avg_scalars[k,:])/np.sqrt(number_of_measurements-1)

#Plot the expectation value as a function of kappa
pl.errorbar(kappas,mean_scalars,yerr=sigma_scalars,fmt="o",markersize=3,color = "black")
pl.xlabel("κ",fontsize=16)
pl.ylabel("⟨ΦΦ⟩",fontsize =16)
pl.show()
