# test run quick routine in  MyForwardModel

#import MyForwardModel
#from myfwd import MyForwardModel
from BayHunter.surf96_modsw import SurfDisp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Input parameters
##h = [10, 20, 30]  # Layer thicknesses
#h = [0.010, 0.020, 0.030]  # Layer thicknesses
##vp = [500, 1000, 2000]  # P-wave velocities
#vp = [5, 10, 20]  # P-wave velocities
##vs = [250, 500, 1000]  # S-wave velocities
#vs = [2.50, 5.00, 10.00]  # S-wave velocities
#rho = [2.400, 2.500, 2.600]  # Densities

# 5 layers
#h = [0.010, 0.020, 0.030, 0.04, 0.05]  # Layer thicknesses
#vp = [5, 10, 15, 20, 22]  # P-wave velocities
#vs = [2.50, 5.00, 7.0, 10.00, 11]  # S-wave velocities
rho = [2.400, 2.500, 2.6, 2.700, 2.7]  # Densities
# new model
#Vs_init=np.array([200, 400, 800, 1500, 2500])
#h_init = np.array([5, 15, 55, 60, 1e3])
vs = [0.2, 0.4, 0.8, 1.5, 2.5]
vp = [0.4, 0.8, 1.6, 3.0, 5.0] 
h = [0.005, 0.015, 0.055, 0.06, 0.05] 

print(vp)
print(vs)

# read disp file
inputfile = "st4_rdispph.dat"
df = pd.read_csv(inputfile, sep=' ', skipinitialspace=True, header=None)
f1 = df[0]
vr1 = df[1]
f_in = np.array(f1)
vr_in = np.array(vr1)
print('freqency(s):',f_in)
print('Rayleigh velocity:', vr_in)


# Initialize model
# generate a numpy array with 50 samples ranging from 0.2 to 20, spaced regularly in logarithmic scale as obsx
#x_obs = np.logspace(np.log10(0.2), np.log10(20), num=50)
#x_obs = np.logspace(np.log10(5), np.log10(20), num=20)
#x_obs = np.logspace(np.log10(0.08), np.log10(0.14), num=20)
x_obs = f_in
print('obsx:',x_obs)
#model = MyForwardModel(obsx=x_obs, ref="test_ref")
model = SurfDisp(obsx=x_obs, ref="rdispph")

## Run the model (includes computation and validation)
xmod, ymod = model.run_model(h, vp, vs, rho)

print("xmod:", xmod)
print("ymod:", ymod)

plt.plot(xmod, ymod, label = 'synthetic')
plt.plot(f_in, vr_in, label = 'data')
plt.legend()
plt.savefig('disp_example.png')
