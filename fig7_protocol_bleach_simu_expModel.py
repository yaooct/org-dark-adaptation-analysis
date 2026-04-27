# -*- coding: utf-8 -*-
"""
Cone Dark Adaptation Protocol & Simulation

Description:
    Generates a two-panel figure for the manuscript:
    - Panel a: Experimental timeline schematic showing the initial bleach 
      and subsequent ORG recordings.
    - Panel b: Simulation of photopigment regeneration, accounting for 
      fractional bleaching caused by sequential probing flashes, overlaid 
      with probabilistic model fitting.

Author: YC
Date: April 2026
"""

import os
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import lambertw

# Attempt to load local publication aesthetics configuration
try:
    import plot_configuration_manuscript as pcfg
    pcfg.setup()
except ImportError:
    print("Warning: 'plot_configuration_manuscript' not found. Using default styles.")
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
        'axes.linewidth': 1.2, 'xtick.direction': 'in', 'ytick.direction': 'in'
    })


# ==========================================
# 1. PARAMETERS & SIMULATION
# ==========================================

# Rate-limited model parameters (Alternative model for reference)
B0 = 0.67
Km = 0.2
v = 0.5 / 60  # s^-1
A_val = (B0 / Km) * np.exp(B0 / Km)
beta = (1 + Km) * v / Km

def p_of_t_RL(t):
    """Rate-limited regeneration model using Lambert W."""
    arg = A_val * np.exp(-beta * t)
    return 1 - Km * lambertw(arg).real

# Standard Exponential Simulation Parameters
T0 = 60.0             # Time constant (s)
B_INIT = 0.67         # Initial bleaching level
B_OFFSET = 0.08       # Offset parameter
TIMES = [20, 40, 60, 90, 130, 150, 180, 300, 400]  # Probe flash times
FINAL_T = 500

# Initialize tracking variables
b_prev = B_INIT
t_prev = 0

all_times, all_b_values, all_frac_b = [0], [B_INIT], [0]
marker_times, marker_values, bfrac = [], [], []

# --- Run Bleach/Regeneration Simulation ---
for t in TIMES:
    # Smooth regeneration curve between flashes
    t_range = np.linspace(t_prev, t, 1000)
    b_curve = b_prev * np.exp(-(t_range - t_prev) / T0)
    
    all_times.extend(t_range)
    all_b_values.extend(b_curve)
    
    # Calculate drop caused by the probe flash
    b = b_prev * np.exp(-(t - t_prev) / T0)
    frac_b = B_OFFSET * (1 - b)
    b_prime = b + frac_b
    
    # Store points for vertical drop lines
    marker_times.extend([t, t])
    marker_values.extend([b, b_prime])
    
    all_frac_b.extend([0] * len(t_range))
    all_frac_b.append(frac_b)
    
    b_prev = b_prime
    t_prev = t
    all_times.append(t)
    all_b_values.append(b_prime)

# Convert marker values to "available pigment"
marker_values = 1 - np.array(marker_values)

# Add final tail decay after the last flash
t_range = np.linspace(t_prev, FINAL_T, 100)
b_curve = b_prev * np.exp(-(t_range - t_prev) / T0)
all_times.extend(t_range)
all_b_values.extend(b_curve)
all_frac_b.extend([0] * len(t_range))


# ==========================================
# 2. PROBABILISTIC CURVE FITTING
# ==========================================
# ==========================================
# 2. MATHEMATICAL MODELS
# ==========================================
def objective_function_helper(tau, eta, Y_eq, Y_start, measurement_times):
    """
    Model: Exponential Increase towards Y_eq + Multiplicative Step Decrease.
    Computes predicted OPL values at the measurement times.
    """
    if tau <= 0: return np.full_like(measurement_times, np.inf, dtype=float)
    
    T_start = measurement_times[0]
    lamb = 1.0 / tau
    values = [Y_start]  # The first fitted point is Y_start
    
    last_time = T_start
    value = Y_start * (1.0 - eta) 
    
    # Iterate from the second measurement time onwards
    for mt in measurement_times[1:]:
        # 1. Exponential Increase/Growth toward Y_eq
        dt = mt - last_time
        value_intermediate = Y_eq - (Y_eq - value) * np.exp(-lamb * dt)        
        
        # 2. Multiplicative Step Decrease occurs *at* the measurement time mt
        value = value_intermediate * (1.0 - eta)         
        
        # The fitted point is the value *before* the step at time mt
        values.append(value_intermediate)                  
        last_time = mt
        
    return np.array(values)


def objective_function(x, tau, eta, Y_eq, Y_initial):
    """Wrapper for curve_fit: fitting four parameters."""
    return objective_function_helper(tau, eta, Y_eq, Y_initial, x)

def yperturb(t_array, tau, eta, Y_eq, Y_initial, measurement_times):
    """
    Continuous-time simulation of an exponential system with stepwise 
    multiplicative perturbations (used for plotting continuous lines).
    """
    if tau <= 0: raise ValueError("tau must be positive.")

    lamb = 1.0 / tau
    T_start = measurement_times[0]
    last_time = T_start
    value = Y_initial
    y_values = []
    step_idx = 0

    for t in t_array:
        dt = t - last_time
        value_continuous = Y_eq - (Y_eq - value) * np.exp(-lamb * dt)

        # Apply steps that occur at or before this time
        while step_idx < len(measurement_times) and t >= measurement_times[step_idx]:
            step_time = measurement_times[step_idx]
            dt_step = step_time - last_time
            value_continuous = Y_eq - (Y_eq - value) * np.exp(-lamb * dt_step)
            value = value_continuous * (1.0 - eta)
            last_time = step_time
            step_idx += 1

        if step_idx == 0 and t < measurement_times[0]:
            y_values.append(Y_eq - (Y_eq - Y_initial) * np.exp(-lamb * (t - T_start)))
        else:
            dt_since_last = t - last_time
            y_values.append(Y_eq - (Y_eq - value) * np.exp(-lamb * dt_since_last))

    return np.array(y_values)

t_fit = TIMES.copy()
opl_fit = marker_values[0::2].copy()  # Grab the pre-flash values

lower_bounds = [30.0, -0.2, 0.0, 0.0] 
upper_bounds = [180.0, 0.2, 1.0, 1.0]
p0 = [60.0, -0.08, 0.02, 0.5]  # [tau, eta, Y_eq, Y0]
bounds = (lower_bounds, upper_bounds)

x = np.array(t_fit)
y = np.array(opl_fit)
t_arr = np.linspace(0, x[-1] + 50, 1000)

res, _ = spo.curve_fit(objective_function, x, y, p0=p0, bounds=bounds, maxfev=50000, ftol=1e-10)
tau_fit, eta_fit, Y_eq_fit, Y_start_fit = res

y_fit = objective_function(x, tau_fit, eta_fit, Y_eq_fit, Y_start_fit)
fiterror = np.sqrt(np.nanmean((y - y_fit) ** 2))

y_fit_arr = yperturb(t_arr, tau_fit, eta_fit, Y_eq_fit, Y_start_fit, x)
Y0_fit = y_fit_arr[0]


# ==========================================
# 3. FIGURE GENERATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7.2, 4))

# Panel Labels
ax1.text(-0.08, 0.85, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax2.text(-0.08, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

# ------------------------------------------
# Panel a: Experimental Timeline Schematic
# ------------------------------------------
ax1.axis('off')
ax1.set_xlim(-1, 12)
ax1.set_ylim(-0.5, 1.8)

# Colors for diagram
COLOR_INIT = '#2ca02c'   
COLOR_PROBE = '#98df8a'  
COLOR_OCT = 'black'

# Base Timeline Arrow
ax1.annotate('', xy=(11.5, 0), xytext=(-1, 0), arrowprops=dict(arrowstyle="-|>", lw=1.2, color='black'))
ax1.text(5.75, -0.15, "Time (s)", ha='center', va='top')

# Initial Bleaching Flash
ax1.add_patch(patches.Rectangle((-1.1, 0), 0.4, 0.8, facecolor=COLOR_INIT, edgecolor='none'))

# Sequenced Probing Flashes & OCT Recording Blocks
blocks = [0, 3.0, 6.0, 8.6]
for i, x_pos in enumerate(blocks):
    ax1.add_patch(patches.Rectangle((x_pos, 0.05), 2.1, 0.4, facecolor='white', edgecolor=COLOR_OCT, lw=1))
    ax1.add_patch(patches.Rectangle((x_pos + 0.3, 0), 0.15, 0.6, facecolor=COLOR_PROBE, edgecolor='none'))

# Ellipsis indicating continued sequence
ax1.text(5.45, 0.25, '...', fontsize=20, fontweight='bold', ha='center', va='center')

# Legend Elements for Panel a
ax1.add_patch(patches.Rectangle((0, 1.3), 0.3, 0.15, facecolor=COLOR_INIT, edgecolor='none'))
ax1.text(0.4, 1.375, r'Initial flash: $3.76 \times 10^7$ photons/$\mu$m$^2$', va='center', ha='left')

ax1.add_patch(patches.Rectangle((0, 1.0), 0.3, 0.15, facecolor=COLOR_PROBE, edgecolor='none'))
ax1.text(0.4, 1.075, r'Probing flash: $2.83 \times 10^6$ photons/$\mu$m$^2$', va='center', ha='left')

ax1.add_patch(patches.Rectangle((5.5, 1.25), 0.6, 0.2, facecolor='white', edgecolor=COLOR_OCT, lw=1))
ax1.text(6.3, 1.35, r'2.1-s OCT volume recording', va='center', ha='left')


# %%
# ------------------------------------------
# Panel b: Bleach Simulation & Fitting
# ------------------------------------------
T0 = 60.0             # Time constant (s)
B_INIT = 0.65         # Initial bleaching level
B_OFFSET = 0.21      # Fractional bleach from probing flashes
TIMES = [20, 40, 60, 90, 130, 180, 230, 300, 400]  # Probe flash times (t_acq)
FINAL_T = 500

# Initialize tracking variables
b_prev = B_INIT
t_prev = 0

all_times, all_b_values = [0], [B_INIT]
marker_times, marker_values = [], []

# --- Run Bleach/Regeneration Simulation ---
for t in TIMES:
    # 1. Smooth regeneration curve between flashes
    t_range = np.linspace(t_prev, t, 500)
    b_curve = b_prev * np.exp(-(t_range - t_prev) / T0)
    
    all_times.extend(t_range)
    all_b_values.extend(b_curve)
    
    # 2. Calculate drop caused by the probe flash
    b_before_flash = b_prev * np.exp(-(t - t_prev) / T0)
    frac_b = B_OFFSET * (1 - b_before_flash)
    b_after_flash = b_before_flash + frac_b
    
    # 3. Store points for vertical drop lines
    marker_times.extend([t, t])
    marker_values.extend([b_before_flash, b_after_flash])
    
    # 4. Update for next iteration
    b_prev = b_after_flash
    t_prev = t
    all_times.append(t)
    all_b_values.append(b_after_flash)

# Add final tail decay after the last flash
t_range = np.linspace(t_prev, FINAL_T, 100)
all_times.extend(t_range)
all_b_values.extend(b_prev * np.exp(-(t_range - t_prev) / T0))

# Convert bleach fractions to "available pigment" (1 - bleach)
marker_values_avail = 1 - np.array(marker_values)
all_avail_pigment = 1 - np.array(all_b_values)

# ------------------------------------------
# Plot Unperturbed Regeneration Curve
# ------------------------------------------
t_baseline = np.arange(FINAL_T)
avail_curve_ideal = 1 - (B_INIT * np.exp(-t_baseline / T0))

ax2.plot(t_baseline, avail_curve_ideal, color='blue', lw=1.5, 
         label=f"Pigment regeneration \n($t_{{0}} = {T0}\\,\\mathrm{{s}}$)")

# ------------------------------------------
# Plot Perturbed Trajectory & Drops
# ------------------------------------------
# Continuous bleaching/regeneration trajectory
ax2.plot(all_times, all_avail_pigment, color='black', linestyle='--', lw=1.2,
         label=f"Simulated remaining \npigment level")

# Fractional bleaching drop lines
for i in range(0, len(marker_times), 2):
    lbl = f'Pigment bleached \nby test flash' if i == 0 else None
    ax2.plot(marker_times[i:i+2], marker_values_avail[i:i+2], color='black', lw=2.5, label=lbl)
    
# Fractional bleaching 
print(f'effective bleach level:')
for i in range(0, len(marker_times), 2):
    print(f'{(marker_values_avail[i]-marker_values_avail[i+1]):.2f}')

# Pre-flash data points
x_data = TIMES
y_data = marker_values_avail[0::2]
ax2.plot(x_data, y_data, marker='*', markersize=12, color='red', 
        linestyle='None', alpha=0.9, label='Measurement')

# ------------------------------------------
# Adaptation Time Conversion
# ------------------------------------------
# Pick a specific acquisition time to highlight (e.g., t_acq = 90s)
t_acq_example = 300
idx = TIMES.index(t_acq_example)
y_adapt = y_data[idx]

ax2.plot([t_acq_example, t_acq_example], [0.2, y_adapt], color='black', linestyle=':', lw=1.5, alpha=0.6)
 
# Calculate the corresponding adaptation time
t_adapt = -T0 * np.log((1 - y_adapt) / B_INIT)
print(f'adaptation time:\n{(-T0 * np.log((1 - y_data) / B_INIT)).round()}s')
# Draw mapping lines
ax2.plot([0, t_acq_example], [y_adapt, y_adapt], color='blue', linestyle=':', lw=1.5, alpha=0.6)
ax2.plot([t_adapt, t_adapt], [0.2, y_adapt], color='blue', linestyle=':', lw=1.5, alpha=0.6)
ax2.plot(t_adapt, y_adapt, marker='o', color='blue', markersize=6)

ax2.text(t_adapt - 5, 0.22, rf'$t_{{adapt}}$' + '\n' + f'({t_adapt:.1f} s)', 
        color='blue', ha='right', va='bottom', fontsize=9)
ax2.text(t_acq_example - 5, 0.22, rf'$t_{{acq}}$' + '\n' + f'({t_acq_example} s)', 
        color='black', ha='right', va='bottom', fontsize=9)

# ------------------------------------------
# Formatting
# ------------------------------------------
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Available Pigment')
ax2.set_xlim(0, FINAL_T)
ax2.set_ylim(0.2, 1.02)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val * 100:.0f}%'))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.05, 0))

# ==========================================
# 4. FINALIZE & SAVE
# ==========================================
plt.tight_layout()

# Uncomment to save the figure automatically
os.makedirs('./figs', exist_ok=True)
plt.savefig('./figs/fig7_protocol_bleach_simu_expModel.png', dpi=600, bbox_inches='tight')

plt.show()