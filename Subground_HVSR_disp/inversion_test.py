import numpy as np
import matplotlib.pyplot as plt
#from Subground_HVSR import *
from Subground_HVSR_disp import *

import pandas as pd

Poisson=0.4
#Vs = np.array([250, 700, 1800, 2500])
##Vp = 1000*(Vs/1000 + 1.164)/0.902
##ro=(310*(Vp*1000)**0.25)/1000
#Vp = Vs * np.sqrt( (1-Poisson)/(0.5 - Poisson))
#ro= 1740 * (Vp/1000)**0.25
#h = np.array([10, 10,17, 1e3])
##Damping
#Dp=np.array( [0.1, 0.05, 0.01, 0.01])
#Ds=np.array( [0.1, 0.05, 0.01,0.01])
#
#f1=0.001
#f2=150.
#freq = np.logspace(np.log10(f1),np.log10(f2),500)
#
#
#mod1 = HVSRForwardModels(fre1=f1,fre2=f2,Ds=Ds,Dp=Dp,h=h,ro=ro,Vs=Vs,Vp=Vp,f=freq,ex=0.0)
#f, hvsr = mod1.HV()
#print("VS",mod1.Vs)

#print("mod1",mod1.Transfer_Function())

# read in hvsr from file
station = "MP01"
#input = pd.read_csv(station+'_hvsr_new.txt', sep=' ', skipinitialspace=True, header=None)
input = pd.read_csv('../'+station+'_hvsr_new.txt', sep=' ', skipinitialspace=True, header=None)
#input = pd.read_csv('../'+station+'_hvsr_20Hz.txt', sep=' ', skipinitialspace=True, header=None)
#input = pd.read_csv('MP01_hvsr_new.txt', sep=' ', skipinitialspace=True, header=None)
#input = pd.read_csv('MP01_hvsr_20Hz.txt', sep=' ', skipinitialspace=True, header=None)
input.head()
f_input = input[0].tolist()
hvsr_input = input[1].tolist()
#print(f_input)
#print(hvsr_input)
f = np.array(f_input)
hvsr = np.array(hvsr_input)
#print(f)
#print(hvsr)

f1 = 0.25
f2 = 50
#f2 = 20
freq = np.logspace(np.log10(f1),np.log10(f2),100)
Dp=np.array( [0.1, 0.05, 0.01, 0.01, 0.01])
Ds=np.array( [0.1, 0.05, 0.01, 0.01, 0.01])


## read disp file
inputfile = "st4_rdispph.dat"
df = pd.read_csv(inputfile, sep=' ', skipinitialspace=True, header=None)
f_in = df[0]
vr_in = df[1]
f_disp = np.array(f_in)
vr_disp = np.array(vr_in)
#print('Dispersion freqency(s):',f_in)
#print('Rayleigh velocity:', vr_in)


## set up initial velocity
# original inital velocity
#Vs_init=np.array([500, 1000, 1500, 2000])
Vs_init=np.array([200, 400, 800, 1500, 2500])
h_init = np.array([5, 15, 55, 60, 1e3])
# velocity from OpenHVSR global inveresion MP01
##Vs_init = np.array([150, 408.15, 806.97, 1388.47, 2500])
##h_init = np.array([5.81, 11.16, 69.37, 59.98, 1e3])
#input2 = pd.read_csv('mean_curve_'+station+'_2022_5month_subsurface.txt', sep=' ', skipinitialspace=True, header=None)
#vs_list = input2[1].tolist()
#h_list = input2[3].tolist()
#Vs_init = np.array(vs_list)
#h_init = np.array(h_list)
#print(Vs_init)
#print(h_init)

Vp_init = Vs_init * np.sqrt( (1-Poisson)/(0.5 - Poisson))
ro_init= 1740 * (Vp_init/1000)**0.25

mod0 = HVSRForwardModels(fre1=f1,fre2=f2,f=freq,Ds=Ds,Dp=Dp,h=h_init,ro=ro_init,Vs=Vs_init,Vp=Vp_init,ex=0.0)
print("VS", mod0.Vs)
f_init, hvsr_init = mod0.HV()

#########################################################################################################
# Amoeba inversion first
#########################################################################################################

#print("mod0",mod0.Transfer_Function())

#print("init",np.c_[f_init,hvsr_init,f,hvsr])

#run1 = HVSR_inversion(hvsr=hvsr,hvsr_freq=f,n=200,n_burn=100,fre1=f1,fre2=f2,Ds=Ds,Dp=Dp,h=h_init,ro=ro_init,Vs=Vs_init,Vp=Vp_init,Vfac=2000,Hfac=50)
run1 = HVSR_inversion(hvsr=hvsr,hvsr_freq=f,n=200,n_burn=100,fre1=f1,fre2=f2,Ds=Ds,Dp=Dp,h=h_init,ro=ro_init,Vs=Vs_init,Vp=Vp_init,Vfac=2000,Hfac=50, disp_freq=f_disp, disp_v=vr_disp)
Vs_best,h_best = run1.Amoeba_crawl()
#print("Amoeba:",results1)


Vp_best = Vs_best * np.sqrt( (1-Poisson)/(0.5 - Poisson))
ro_best= 1740 * (Vp_best/1000)**0.25

#Vp_best=1000*(Vs_best/1000 + 1.164)/0.902
#ro_best=(310*(Vp_best*1000)**0.25)/1000

h1=h_best
mod2 = HVSRForwardModels(fre1=f1,fre2=f2,f=freq,Ds=Ds,Dp=Dp,h=h_best,ro=ro_best,Vs=Vs_best,Vp=Vp_best,ex=0.0)
hvsr_best_f, hvsr_best = mod2.HV()

#  End amoeba, create new model for MCMC 
###################################################################################################
#print("Results",np.shape(results))
