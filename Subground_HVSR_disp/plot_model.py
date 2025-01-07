import numpy as np
import matplotlib.pyplot as plt
from Subground_HVSR import *
#from Subground_HVSR_disp import *

import pandas as pd

station = 'MP01'
## read in files 
input1 = "velocity_model_original.csv"
df1 = pd.read_csv(input1)
#print(df1.head())
VS0 = df1['vs'].tolist()
D0 = df1['h'].tolist()
print(VS0)
print(D0)

input2 = "velocity_model_nm.csv"
df2 = pd.read_csv(input2)
#print(df2.head())
VS5 = df2['vs'].tolist()
D5 = df2['h'].tolist()
print(VS5)
print(D5)

input3 = "velocity_model_mcmc.csv"
df3 = pd.read_csv(input3)
#print(df3.head())
VS2 = df3['vs'].tolist()
D2 = df3['h'].tolist()
print(VS2)
print(D2)

## generate velocity model from Marine
Poisson=0.4
h_new = np.array([ 5, 12, 5])
Vs_new = np.array([ 275, 550, 750])
Vp_new = Vs_new * np.sqrt( (1-Poisson)/(0.5 - Poisson))
ro_new= 1740 * (Vp_new/1000)**0.25
p6p = HVSR_plotting_functions(h=h_new,ro=ro_new,Vs=Vs_new,Vp=Vp_new)
VS6,D6 = p6p.profile_4_plot()

## plot
plt.figure( dpi=450)
plt.clf()
plt.rcParams['axes.linewidth'] = 1.2

plt.plot(VS0,D0,color="xkcd:midnight blue",linewidth=0.8,label="Initial Condition")
plt.plot(VS5,D5,color="xkcd:blood red",label="Inversion NM",linestyle="--")
plt.plot(VS2,D2,color="xkcd:kelly green",label="Inversion MCMC",linestyle="--")
plt.plot(VS6,D6,color="xkcd:blue",label="Inversion Dispersion",linestyle="--")
plt.xlabel('Vs (m/s)', fontsize=16)
plt.ylabel('Depth (m)', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.ylim(np.max(D[-3])+40,0)
plt.ylim(np.max(D2[-3])+40,0)
#plt.xlim(np.min(VS)-50,np.max(VS)+50)
#plt.savefig("model2.png")
#plt.savefig("model2_"+station+".png")
#plt.savefig("model2_test3_"+station+".png")
plt.savefig("model2_test6_"+station+".png", dpi=450, bbox_inches='tight')

