import sesame
from sesame.builder import Builder
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

#?
# -----------------------------------------------------------------------------
# System definition
# -----------------------------------------------------------------------------
L = 1e-3  # length of the system [cm]

# Mesh
x = np.concatenate((np.linspace(0, 0.4e-3, 400, endpoint=False),
                    np.linspace(0.41e-3, 0.6e-3, 900),
                    np.linspace(0.61e-3, L, 400)))

sys = sesame.Builder(x)

# Material parameters
material = {
        'Nc':1.04e19,
        'Nv':6.0e18,
        'Eg':0.66,
        'affinity':4.0,
        'epsilon':16,
        'mu_e':3900,
        'mu_h':1900,
#Shockley Read Hall Recomb
        'tau_e':1e-6,
        'tau_h':1e-6,
        'Et':0,        
#Auger recomb
        'Cn':1e-31,
        'Cp':1e-31
}
sys.add_material(material)

# Junction definition
junction = L / 2

def n_region(pos): return pos < junction
def p_region(pos): return pos >= junction

# Doping
sys.add_donor(1e18, n_region)
sys.add_acceptor(1e16, p_region)

# Contacts
sys.contact_type('Ohmic', 'Ohmic')
sys.contact_S(1e7, 0, 0, 1e7)

# Generation (none here)
phi = 0
alpha = 4.3e4
def gfcn(x, y): return phi * alpha * np.exp(-alpha * x)
sys.generation(gfcn)

# -----------------------------------------------------------------------------
# I–V sweep (includes both forward and reverse bias)
# -----------------------------------------------------------------------------
Vmin, Vmax = -5, 2  # volts
voltages = np.linspace(Vmin, Vmax, 50)

# Compute current density from SESAME
j = sesame.IVcurve(sys, voltages, '1dhomo_V', verbose=True)
j = j * sys.scaling.current  # convert to physical units [A/cm²]

# Save data
result = {'v': voltages, 'j': j}
np.save('IV_values.npy', result)
np.savetxt('IV_values.txt', np.column_stack((voltages, j)))
savemat('IV_values.mat', result)

# -----------------------------------------------------------------------------
# Shockley diode analytical curve for comparison
# -----------------------------------------------------------------------------
def shockley_diode(V, Is=10e-6, n=1, T=300):
    k = 1.380649e-23
    q = 1.602176634e-19
    Vt = k * T / q
    return Is * (np.exp(V / (n * Vt)) - 1)

V_shockley = np.linspace(Vmin, Vmax, 400)
I_shockley = shockley_diode(V_shockley, Is=10e-6, n=1, T=300)

# -----------------------------------------------------------------------------
# Combined plot (forward + reverse bias)
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.semilogy(V_shockley, np.abs(I_shockley), 'r--', lw=2,
             label='Shockley Equation')
plt.semilogy(voltages, np.abs(j), 'go-', lw=2,
             label='SESAME Simulation')

plt.xlabel('Voltage (V)')
plt.ylabel('|Current| (A/cm²)')
plt.title('Diode I–V Curve (Forward + Reverse Bias)')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
