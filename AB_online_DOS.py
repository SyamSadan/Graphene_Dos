import matplotlib
matplotlib.rcParams['text.usetex'] = True
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import random as rd


#Lattice Parameter

a=142*(np.sqrt(3))  # (10**(-12))

# Parameters

g0=3.16        # in eV
g1=0.381       # in eV
g3=0.38        # in eV
g4= 0.14       # in eV
epsA1=0.022    # in eV
epsA2=0.022    # in eV
epsB1=0        # in eV
epsB2=0        # in eV
V=0            # in eV
Xi=1           # Valley Degeneracy Quantum no


def f(kx,ky):
    func=( np.exp(1j*ky*a/(np.sqrt(3))) ) + ( 2*np.cos(kx*a/(2))*(np.exp((-1j)*ky*a/(2*np.sqrt(3)))) )
    return func
f1= lambda kx,ky: np.conj(f(kx,ky))



def eigen(brill):
    # Matrix Elements
    kx=brill[0]
    ky=brill[1]
    H11=epsA1
    H12=-g0*f(kx,ky)
    H13=g4*f(kx,ky)
    H14=-g3*f1(kx,ky)

    H21=-g0*f1(kx,ky)
    H22=epsB1
    H23=g1
    H24=g4*f(kx,ky)

    H31=g4*f1(kx,ky)
    H32=g1
    H33=epsA2
    H34=-g0*f(kx,ky)

    H41=-g3*f(kx,ky)
    H42=g4*f1(kx,ky)
    H43=-g0*f1(kx,ky)
    H44=epsB2

    x = np.array([[H11, H12, H13, H14],[H21,H22,H23,H24],[H31,H32,H33, H34],[H41, H42, H43, H44]])
    Q=Qobj(x)
    En=Q.eigenenergies()
    val=0
    Energy1=[] # lowest band valance
    Energy2=[] # second lowest valance
    Energy3=[] # lowest conduction band
    Energy4=[] # highest conduction band
    while val==0:
            Energy1.append(En[val])
            Energy2.append(En[val+1])
            Energy3.append(En[val+2])
            Energy4.append(En[val+3])
            val+=1
    return En 

def Energy_AB(start):
    Kx=np.linspace(-0.025,0.025,1000)
    Ky=np.zeros_like(Kx)
    Brill=[]
    for i in range(len(Kx)):
        Brill.append([Kx[i],Ky[i]])
    for l in range(len(Brill)):
            brill=Brill[l]
            eigen(brill)
    return Energy1, Energy2, Energy3, Energy4





b=142e-12 # meter
Area=3*np.sqrt(3)*(b**2)/(2)
# Corners of Brillouin Zone
factor= 4*np.pi/(b*3*np.sqrt(3))
x1 = factor * (-1/2)
y1 = factor * (-np.sqrt(3)/2)

x2 = factor * (-1)
y2 = factor * (0)

x3 = factor * (-1/2)
y3 = factor * (np.sqrt(3)/2)

x4 = factor * (1/2)
y4 = factor * (np.sqrt(3)/2)

x5 = factor * (1)
y5 = factor * (0)

x6 = factor * (1/2)
y6 = factor * (-np.sqrt(3)/2)

m1 = ( y1 - y2 )/( x1 - x2 )
m2 = ( y2 - y3 )/( x2 - x3 )
m3 = ( y4 - y5 )/( x4 - x5 )
m4 = ( y6 - y5 )/( x6 - x5 ) 

coordinates=[ [x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6] ]

################################################################################
# Auxilliary functions

def Condition(kx,ky):
    Y1 = ky >= m1*(kx-x1)+y1 
    Y2 = ky <= m2*(kx-x2)+y2
    Y3 = ky <= m3*(kx-x4)+y4
    Y4 = ky >= m4*(kx-x6)+y6
    if Y1 and Y2 and Y3 and Y4:
        Bool=True
    else :
        Bool=False
    return Bool


def gaussian(Energy,Eigen_value,sigma):
    norm = 1/(sigma*np.sqrt(2*np.pi))
    expon = (-1/2) * ( (Energy-Eigen_value)**2 /(sigma**2) )
    func = norm*np.exp(expon)
    return func



################################################################################

# Defining functions for plotting
################################################################################
# Brillouin Zone image
def Brillouin(give_any_no):
    xval=[]
    yval=[]
    for i in range(0,6):
        xval.append(coordinates[i][0])
        yval.append(coordinates[i][1])
        plt.scatter(coordinates[i][0],coordinates[i][1],color='r')
    xval.append(coordinates[0][0])
    yval.append(coordinates[0][1])
    plt.plot(xval,yval,color='r')        
    plt.show()
    plt.close()

def Brill(N):
    lim1x = coordinates[1][0]
    lim2x = coordinates[4][0]
    lim1y = coordinates[0][1]
    lim2y = coordinates[2][1]
    Accepted = 0
    Brill=[]
    Kx=[]
    Ky=[]
    for j in range(int(N)+1):
        kx=rd.uniform(lim1x,lim2x)
        ky=rd.uniform(lim1y,lim2y)
        Bool=Condition(kx,ky)
        if Bool:
            points=[kx,ky]
            Kx.append(kx)
            Ky.append(ky)
            Brill.append(points)        
            Accepted+=1
    return Brill,Kx,Ky

def PlotBrill(N):   
    xval=[]
    yval=[]
    brill,kx,ky = Brill(N)
    for i in range(0,6):
        xval.append(coordinates[i][0])
        yval.append(coordinates[i][1])
        plt.scatter(coordinates[i][0],coordinates[i][1],color='r')
    xval.append(coordinates[0][0])
    yval.append(coordinates[0][1])
    plt.plot(xval,yval,color='r')        
    plt.plot(kx,ky,',')
    plt.show()
    plt.close()
################################################################################


# Defining functions to find the density of states.
################################################################################
#Density of states
def Denity_Levels(eigen):
    N=1e6      # No of Brillouin points taken to find the density of states
    M=1000     # No of energy levels chosen for plotting the DOS from Emin to Emax
    dE=1e-2    # the energy interval
    n_runs = 1 # For averaging over n runs
    Volume = 1
    Emax=max(eigen([0,0]))
    Energy_list = np.array([ i  for i in np.linspace(-Emax,Emax,M) ])    
    Counts_hvlc=[0]*len(Energy_list)
    Counts_lvhc=[0]*len(Energy_list)
    for run in range(n_runs):
        brill,kx,ky=Brill(N)
        Eigen_list_hvlc = []    # hv-highest valance, lc-lowest conduction
        Eigen_list_lvhc = []
        for i in range(len(brill)):
            e_lv,e_hv,e_lc,e_hc=eigen(brill[i])
            Eigen_list_hvlc.append(e_lc)
            Eigen_list_hvlc.append(e_hv)
            Eigen_list_lvhc.append(e_lv)
            Eigen_list_lvhc.append(e_hc)
        Eigen_list_hvlc=np.array(Eigen_list_hvlc)
        Eigen_list_lvhc=np.array(Eigen_list_lvhc)
        for i in range(len(Energy_list)):
            count=0
            count1=0
            for eigen0 in Eigen_list_hvlc:
                if Energy_list[i]<=eigen0 and eigen0<= (Energy_list[i]+dE):
                    count+=1
            Counts_hvlc[i]+=count
            for eigen1 in Eigen_list_lvhc:
                if Energy_list[i]<=eigen1 and eigen1<= (Energy_list[i]+dE):
                    count1+=1
            Counts_lvhc[i]+=count1
    Counts_lvhc=np.array(Counts_lvhc)*(1/n_runs)*(1/Volume)*(1/max(Counts_lvhc))
    Counts_hvlc=np.array(Counts_hvlc)*(1/n_runs)*(1/Volume)*(1/max(Counts_hvlc))
    plt.plot(Energy_list,Counts_lvhc,color='blue',label='Bands-1&4')
    plt.plot(Energy_list,Counts_hvlc,color='red',label='Bands-2&3')    
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of energy levels')
    plt.xlim(-Emax,Emax)
    plt.legend()
    plt.savefig('ABDOSG_Count.svg')
    plt.savefig('AB6DOSG_Count.png')
    plt.show()
    plt.close()
#Density of states
def Delta_method(eigen): 
    N=1e6
    M=1000
    sigma=1e-4
    Emax=max(eigen([0,0]))
    Energy_list = np.array([ i  for i in np.linspace(-Emax,Emax,M) ])  
    GE1=[]
    GE2=[]
    brill,kx,ky=Brill(N)
    Eigen_list_hvlc = []    # hv-highest valance, lc-lowest conduction
    Eigen_list_lvhc = []
    for i in range(len(brill)):
        e_lv,e_hv,e_lc,e_hc=eigen(brill[i])
        Eigen_list_hvlc.append(e_lc)
        Eigen_list_hvlc.append(e_hv)
        Eigen_list_lvhc.append(e_lv)
        Eigen_list_lvhc.append(e_hc)
    Eigen_list_hvlc=np.array(Eigen_list_hvlc)
    Eigen_list_lvhc=np.array(Eigen_list_lvhc)
    for i in range(len(Energy_list)):
        sum1=0
        sum2=0
        E=Energy_list[i]
        for eigen1 in Eigen_list_hvlc:
            sum1+=gaussian(E,eigen1,sigma)
        GE1.append(sum1)
        for eigen2 in Eigen_list_lvhc:
            sum2+=gaussian(E,eigen2,sigma)
        GE2.append(sum2)
    GE1=np.array(GE1)
    GE2=np.array(GE2)
    plt.plot(Energy_list,GE1,color='red',label='Bands-2&3')
    plt.plot(Energy_list,GE2,color='blue',label='Bands-1&4')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of energy levels')
    plt.xlim(-Emax,Emax)
    plt.legend()
    plt.savefig('AB1DOSG1_delta.svg')
    plt.savefig('AB1DOSG1_delta.png')
    plt.show()
    plt.close()    
 


Delta_method(eigen)
Denity_Levels(eigen)
