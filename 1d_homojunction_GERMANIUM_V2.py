import sesame
from sesame.builder import Builder
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

# %%=============================================================================
# 1. Physical Parameters
# =============================================================================

# SRH lifetimes [s]
tau_e = 1e30 #1-100mu s
tau_h = tau_e
Et = 0       #0.1-0.3eV
# Auger coefficients [cm^6/s]
Cn = 1e-29   #5e-32 - 1e-31
Cp = 0    #1e-31 - 5e-31

B = 0 ## radiation recombination 1e-14 - 1e-12

## For = recomb @0.3V tau =1e-7 B =5e-11 C= 2e-29

#%% =============================================================================
# 2. System Definition
# =============================================================================

L = 1e-3  # device length [cm]

# Non-uniform mesh
x = np.concatenate((
    np.linspace(0, 0.4e-3, 400, endpoint=False),
    np.linspace(0.41e-3, 0.6e-3, 900),
    np.linspace(0.61e-3, L, 400)
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
sys.add_acceptor(1e18, p_region)

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

Vmin, Vmax = -5, 0.9
voltages = np.linspace(Vmin, Vmax, 100)

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
plt.grid(False)
plt.legend(loc='best')  # Auto-place legend
plt.tight_layout()
plt.show()


# %%
#8. Energy Band Diagrams (SESAME Analyzer)

# ---- Choose voltages you want to visualize ----
voltages_to_plot = [0.0, 0.4]   # equilibrium, forward, reverse

for Vtarget in voltages_to_plot:
    
    # Find closest simulated voltage index
    idx = np.argmin(np.abs(voltages - Vtarget))
    Vactual = voltages[idx]
    
    print(f"Loading band diagram for V = {Vactual:.3f} V (index {idx})")
    
    # Load saved simulation
    sys_loaded, result_loaded = sesame.load_sim(f'1dhomo_V_{idx}.gzip')
    
    # Create analyzer
    az = sesame.Analyzer(sys_loaded, result_loaded)
    
    # Define line across device 
    p1 = (0, 0)
    p2 = (L, 0)
    
    # Plot band diagram
    az.band_diagram((p1, p2))
    
    plt.title(f'Energy Band Diagram (V = {Vactual:.2f} V)')

plt.tight_layout()
plt.show()


# %% ===========================================================================
# 9. Voltage and Current Profiles Across n and p Regions
# =============================================================================

# --- Choose voltage to analyze ---
Vtarget = 0.3  # V
idx = np.argmin(np.abs(voltages - Vtarget))
Vactual = voltages[idx]

print(f"\nAnalyzing voltage and currents at V = {Vactual:.3f} V")

# --- Load saved simulation ---
sys_loaded, result_loaded = sesame.load_sim(f'1dhomo_V_{idx}.gzip')
az = sesame.Analyzer(sys_loaded, result_loaded)

# --- Define line across device ---
p1 = (0, 0)
p2 = (L, 0)
x_rel, sites = az.line(sys_loaded, p1, p2)

# --- Absolute position in µm ---
x_um = sys_loaded.xpts[sites] * 1e4  # convert cm → µm

# --- Electrostatic potential in volts ---
V_dimless = result_loaded['v'][sites]
V = V_dimless * sys_loaded.scaling.energy  # convert to volts

# --- Junction position ---
junction_um = (L/2) * 1e4

# --- Separate n and p regions ---
n_indices = x_um < junction_um
p_indices = x_um >= junction_um

# --- Voltage drops ---
Vn_drop = V[n_indices][-1] - V[n_indices][0]
Vp_drop = V[p_indices][-1] - V[p_indices][0]
Vtotal_internal = V[-1] - V[0]

print(f"Voltage drop across n-region  = {Vn_drop:.4f} V")
print(f"Voltage drop across p-region  = {Vp_drop:.4f} V")
print(f"Total internal voltage drop   = {Vtotal_internal:.4f} V")

# --- Plot electrostatic potential ---
plt.figure(figsize=(8,5))
plt.plot(x_um, V, 'b-', lw=2, label='Potential')
plt.axvline(junction_um, color='k', linestyle='--', label='Junction')
plt.xlabel("Position (µm)")
plt.ylabel("Potential (V)")
plt.title(f"Voltage Profile at {Vactual:.2f} V")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# --- Electron and hole currents along the line ---
Jn = az.electron_current(location=(p1,p2))
Jp = az.hole_current(location=(p1,p2))

# --- Interpolate to same x positions for plotting ---
# az.line returns sites, currents are one element shorter
x_current_um = x_um[:-1]

plt.figure(figsize=(8,5))
plt.plot(x_current_um, Jn*1e3, 'r-', lw=2, label='Electron Current (mA/cm²)')
plt.plot(x_current_um, Jp*1e3, 'g-', lw=2, label='Hole Current (mA/cm²)')
plt.axvline(junction_um, color='k', linestyle='--', label='Junction')
plt.xlabel("Position (µm)")
plt.ylabel("Current Density (mA/cm²)")
plt.title(f"Electron and Hole Currents at {Vactual:.2f} V")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

#%%
# %% ===========================================================================
# 9. Voltage and Total Current Profile
# ============================================================================

# --- Choose voltage to analyze ---
Vtarget = 0.3  # Voltage to inspect
idx = np.argmin(np.abs(voltages - Vtarget))
Vactual = voltages[idx]
print(f"\nAnalyzing profiles at V = {Vactual:.3f} V")

# --- Load saved simulation ---
sys_loaded, result_loaded = sesame.load_sim(f'1dhomo_V_{idx}.gzip')
az = sesame.Analyzer(sys_loaded, result_loaded)

# --- Define line across device ---
p1 = (0, 0)
p2 = (L, 0)
x, sites = az.line(sys_loaded, p1, p2)

# --- Extract electrostatic potential (dimensionless → V) ---
V_dimless = az.v[sites]
V_profile = V_dimless * sys_loaded.scaling.energy  # Convert to volts

# --- Define junction position in µm ---
junction_um = (L/2) * 1e4

# --- Separate n and p regions ---
n_mask = x_um < junction_um
p_mask = x_um >= junction_um

# --- Voltage drops ---
Vn_drop = V_profile[n_mask][0] - V_profile[n_mask][-1]
Vp_drop = V_profile[p_mask][0] - V_profile[p_mask][-1]

print(f"Voltage drop across n-region = {Vn_drop:.4f} V")
print(f"Voltage drop across p-region = {Vp_drop:.4f} V")
print(f"Total internal voltage drop  = {Vn_drop + Vp_drop:.4f} V")

# --- Compute currents ---
Jn = az.electron_current(location=(p1, p2))  # A/cm²
Jp = az.hole_current(location=(p1, p2))      # A/cm²

J_total = Jn + Jp                             # Total current density
J_total_mA = J_total * 1e3                    # Convert to mA/cm²

# --- Plot voltage profile ---
plt.figure(figsize=(8,5))
plt.plot(x_um, V_profile, 'r-', lw=2, label='Electrostatic Potential (V)')
plt.axvline(junction_um, color='k', linestyle='--', label='Junction')
plt.xlabel("Position (µm)")
plt.ylabel("Voltage (V)")
plt.title(f"Voltage Profile at V = {Vactual:.2f} V")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot total current ---
plt.figure(figsize=(8,5))
plt.plot(x_um[:-1], J_total_mA, 'b-', lw=2, label='Total Current (mA/cm²)')
plt.axvline(junction_um, color='k', linestyle='--', label='Junction')
plt.xlabel("Position (µm)")
plt.ylabel("Current (mA/cm²)")
plt.title(f"Total Current Profile at V = {Vactual:.2f} V")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()
