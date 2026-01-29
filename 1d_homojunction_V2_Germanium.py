import sesame
from sesame.builder import Builder
import matplotlib.pyplot as plt
#%%

import numpy as np

L = 1e-3 # length of the system in the x-direction [cm]

# Mesh
x = np.concatenate((np.linspace(0,0.4e-3, 120, endpoint=False),
                    np.linspace(0.401e-3,0.6e-3,300),
                    np.linspace(0.601e-3, L, 120)))

# Create a system
sys = sesame.Builder(x)
#%%
# Dictionary with the material parameters
material_GE = {'Nc':1.04e19,
            'Nv':6.0e18,
            'Eg':0.66,
            'affinity':4.0,
            'epsilon':16,
        'mu_e':3900,
        'mu_h':1900,
        'tau_e':1e-6,
        'tau_h':1e-6,
        'Et':0}

# Add the material to the system
sys.add_material(material_GE)

junction = L/2 # extent of the junction from the left contact [m]

def n_region(pos):
    x = pos
    return x < junction

def p_region(pos):
    x = pos
    return x >= junction

# Add the donors
nD = 1e18 # [cm^-3]
sys.add_donor(nD, n_region)

# Add the acceptors
nA = 1e16 # [cm^-3]
sys.add_acceptor(nA, p_region)

# Define Ohmic contacts
sys.contact_type('Ohmic', 'Ohmic')

# Define the surface recombination velocities for electrons and holes [cm/s]
Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 0, 0, 1e7
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

# Define a function for the generation rate
phi = 0         # photon flux [1/(cm^2 s)] 1e17
alpha = 4.3e4      # absorption coefficient [1/cm]

# Define a function for the generation rate
def gfcn(x,y):
    return phi * alpha * np.exp(-alpha * x)

# add generation to system
sys.generation(gfcn)

# IV curve
Vmin, Vmax = -2, 0 # Voltage range (V)
voltages = np.linspace(Vmin, Vmax, 30)
j = (sesame.IVcurve(sys, voltages, '1dhomo_V', verbose =True)*-1)

# convert dimensionless current to dimension-ful current
j = j * sys.scaling.current
# save voltage and current values to dictionary
result = {'v':voltages, 'j':j}

# save data to python data file
np.save('IV_values', result)

# save data to an ascii txt file
np.savetxt('IV_values.txt', (voltages, j))



#%%
def shockley_diode(V, Is=3e-14, n=1, T=300):
    
    k = 1.380649e-23   # Boltzmann constant (J/K)
    q = 1.602176634e-19  # Electron charge (C)
    Vt = k * T / q       # Thermal voltage (V)
    
    return Is * (np.exp(V / (n * Vt)) - 1)

# ---- Parameters ----
Is = 1e-8       # Saturation current (A)
n = 1        # Ideality factor
T = 300          # Temperature (K)
Vmin, Vmax = -5, 0 # Voltage range (V)

# ---- Voltage array ----
V = np.linspace(Vmin, Vmax, 100)

# ---- Compute current ----
I = (shockley_diode(V, Is, n, T)) 

# save data to a matlab data file


# ---- Plot ----
plt.figure(figsize=(8, 6))
plt.plot(V, I, label=f"Shockley equation (n={n}, T={T}K)",color = "red")
plt.yscale("log")  # Log scale to show exponential behavior
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("Shockley equation (red) vs Sesame (green)")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.show()
#%%

from scipy.io import savemat
savemat('IV_values.mat', result)

# plot I-V curve


plt.plot(voltages, j,'-o',label=f"Sesame (n={n}, T={T}K)",color = "green")
plt.xlabel('Voltage [V]')
plt.yscale("log")
plt.ylabel('Current [A/cm^2]')
plt.grid()     # add grid
plt.show()     # show the plot on the screen
