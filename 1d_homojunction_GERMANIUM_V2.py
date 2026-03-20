import sesame
from sesame.builder import Builder
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

# %%=============================================================================
# 1. Physical Parameters
# =============================================================================

# SRH lifetimes [s]
tau_e = 1e-7 #1-100mu s
tau_h = tau_e
Et = 0.15      #0.1-0.3eV
# Auger coefficients [cm^6/s]
Cn = 5e-31   #5e-32 - 1e-31
Cp = Cn    #1e-31 - 5e-31

B = 1e-12 ## radiation recombination 1e-14 - 1e-12

## For = recomb @0.3V tau =1e-7 B =5e-11 C= 2e-29

#%% =============================================================================
# 2. System Definition
# =============================================================================

L = 1e-1 # device length [cm]

# Non-uniform mesh
x = np.concatenate((
    np.linspace(0, 0.45e-1, 400, endpoint=False),##########kkkjhg
    np.linspace(0.451e-1, 0.55e-1, 900),
    np.linspace(0.551e-1, L, 400)
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

Vmin, Vmax = 0, 0.6
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
#8. Energy Band Diagrams

# ---- Choose voltages you want to visualize ----
voltages_to_plot = [0.4, 0.6, 0.1]   # equilibrium, forward, reverse

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


# %% 9. Voltage and Current Profiles Across n and p Regions

# --- Choose voltage to analyze ---
Vtarget = 0  # V
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
Vn_drop = V[n_indices][0] - V[n_indices][-1]
Vp_drop = V[p_indices][0] - V[p_indices][-1]

print(f"Voltage drop across n-region  = {Vn_drop:.4f} V")
print(f"Voltage drop across p-region  = {Vp_drop:.4f} V")
print(f"Total internal voltage drop  = {Vn_drop + Vp_drop:.4f} V")

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


#%% 10. Electron and Hole Density Across Device
# 11. Quasi-Fermi Levels Across the Junction



# --- Load the simulation for the voltage you want ---
Vtarget = 0# V
idx = np.argmin(np.abs(voltages - Vtarget))
Vactual = voltages[idx]

print(f"Requested voltage = {Vtarget:.2f} V")
print(f"Using simulated voltage = {Vactual:.3f} V")

sys_loaded, result_loaded = sesame.load_sim(f'1dhomo_V_{idx}.gzip')
az = sesame.Analyzer(sys_loaded, result_loaded)

# --- Define line across the device ---
p1 = (0, 0)
p2 = (L, 0)
x_rel, sites = az.line(sys_loaded, p1, p2)
x_um = sys_loaded.xpts[sites] * 1e4  # convert cm → µm
junction_um = (L/2)*1e4

# --- Electron and hole densities in physical units ---
n = az.electron_density(location=(p1, p2)) * sys_loaded.scaling.density
p = az.hole_density(location=(p1, p2)) * sys_loaded.scaling.density

# --- Plot electron and hole densities ---
plt.figure(figsize=(8,5))
plt.plot(x_um, n, 'r-', lw=2, label='Electron Density (n, cm⁻³)')
plt.plot(x_um, p, 'g-', lw=2, label='Hole Density (p, cm⁻³)')
plt.axvline(junction_um, color='k', linestyle='--', label='Junction')
plt.xlabel("Position (µm)")
plt.ylabel("Carrier Density (cm⁻³)")
plt.title(f"Electron and Hole Density at V = {Vactual:.2f} V")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# --- Quasi-Fermi levels in eV ---
Efn = result_loaded['efn'][sites] * sys_loaded.scaling.energy
Efp = result_loaded['efp'][sites] * sys_loaded.scaling.energy

plt.figure(figsize=(8,5))
plt.plot(x_um, Efn, 'r-', lw=2, label='Electron Quasi-Fermi Level (Efn)')
plt.plot(x_um, Efp, 'b-', lw=2, label='Hole Quasi-Fermi Level (Efp)')
plt.axvline(junction_um, color='k', linestyle='--', label='Junction')
plt.xlabel("Position (µm)")
plt.ylabel("Energy (eV)")
plt.title(f"Quasi-Fermi Levels at V = {Vactual:.2f} V")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()