import sesame
from sesame.builder import Builder
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

# %%=============================================================================
# 1. Physical Parameters
# =============================================================================

# SRH lifetimes [s]
tau_e = 1e-6  #1-100mu s
tau_h = 1e-6
Et = 0       #0.1-0.3eV
# Auger coefficients [cm^6/s]
Cn = 0    #5e-32 - 1e-31
Cp = 0    #1e-31 - 5e-31

B = 1e-14 ## radiation recombination 1e-14 - 1e-12

#%% =============================================================================
# 2. System Definition
# =============================================================================

L = 1e-3  # device length [cm]

# Non-uniform mesh
x = np.concatenate((
    np.linspace(0, 0.4e-3, 40, endpoint=False),
    np.linspace(0.41e-3, 0.6e-3, 90),
    np.linspace(0.61e-3, L, 40)
))

sys = Builder(x)

# %%=============================================================================
# 3. Material Parameters
# =============================================================================

material = {
    'Nc': 1.04e19,
    'Nv': 6.0e18,
    'Eg': 0.66,
    'affinity': 4.0,
    'epsilon': 16,
    'mu_e': 3900,
    'mu_h': 1900,

    # SRH recombination (lifetime model)
    'tau_e': tau_e,
    'tau_h': tau_h,
    'Et': Et,      # trap level at midgap

    # Auger recombination
    'Cn': Cn,
    'Cp': Cp,
    'B' : B
}

sys.add_material(material)

# %%=============================================================================
# 4. Junction Definition
# =============================================================================

junction = L / 2

def n_region(pos): return pos < junction
def p_region(pos): return pos >= junction

# Doping
sys.add_donor(1e18, n_region)
sys.add_acceptor(1e16, p_region)

# Contacts
sys.contact_type('Ohmic', 'Ohmic')
# Define the surface recombination velocities for electrons and holes [cm/s]
Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 0, 0, 1e7  # cm/s
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

# Dark simulation (no optical generation)
phi = 0
alpha = 4.3e4
def gfcn(x, y): return phi * alpha * np.exp(-alpha * x)
sys.generation(gfcn)

# %%=============================================================================
# 5. I–V Sweep
# =============================================================================

Vmin, Vmax = -5, 2
voltages = np.linspace(Vmin, Vmax, 50)

j = sesame.IVcurve(sys, voltages, '1dhomo_V', verbose=True)
j = j * sys.scaling.current  # Convert to A/cm²

# Save IV data
result = {'v': voltages, 'j': j}
np.save('IV_values.npy', result)
np.savetxt('IV_values.txt', np.column_stack((voltages, j)))
savemat('IV_values.mat', result)

# =============================================================================
# %%6. Shockley Analytical Comparison
# =============================================================================

def shockley_diode(V, Is=1e-5, n=1, T=300):
    k = 1.380649e-23
    q = 1.602176634e-19
    Vt = k * T / q
    return Is * (np.exp(V / (n * Vt)) - 1)

V_shockley = np.linspace(Vmin, Vmax, 400)
I_shockley = shockley_diode(V_shockley)

# =============================================================================
# %%7. Plot I–V Curve with recombination info
plt.figure(figsize=(9, 6))

# Shockley curve
plt.semilogy(V_shockley, np.abs(I_shockley), 'r--', lw=2,
             label='Shockley Equation')

# SESAME simulation curve with recombination values each on its own line
sesame_label = (
    'SESAME Simulation\n'
    f'τe = {tau_e:.1e} s\n'
    f'τh = {tau_h:.1e} s\n'
    f'Et = {Et} eV\n'
    f'Cn = {Cn:.1e} cm⁶/s\n'
    f'Cp = {Cp:.1e} cm⁶/s\n'
    f'B = {B:.1e} cm³/s'
)

plt.semilogy(voltages, np.abs(j), 'go-', lw=2, label=sesame_label)

plt.xlabel('Voltage (V)')
plt.ylabel('|Current| (A/cm²)')
plt.title('Diode I–V Curve (Forward + Reverse Bias)')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend(loc='best')  # Auto-place legend
plt.tight_layout()
plt.show()

