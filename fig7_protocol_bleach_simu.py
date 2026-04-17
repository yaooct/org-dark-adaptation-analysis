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


# ------------------------------------------
# Panel b: Bleach Simulation & Fitting
# ------------------------------------------
t_baseline = np.arange(FINAL_T)

# Unperturbed regeneration curve
b_curve_model = B_INIT * np.exp(-t_baseline / T0)
ax2.plot(t_baseline, 1 - b_curve_model, color='blue', lw=1.2, 
         label=f"Pigment regeneration ($t_{{0}} = {T0}\\,\\mathrm{{s}}$)")

# Fractional bleaching drop lines
for i in range(0, len(marker_times), 2):
    lbl = r'Pigment bleached by test flash' if i == 0 else None
    ax2.plot(marker_times[i:i+2], marker_values[i:i+2], color='black', lw=2.5, label=lbl)
    bfrac.append(marker_values[i+1] - marker_values[i])

# Continuous bleaching/regeneration trajectory
ax2.plot(all_times, 1 - np.array(all_b_values), color='black', linestyle='--', lw=1.2,
         label="Simulated remaining pigment level")

# Data vs Fit Visualization
ax2.plot(x, y, marker='*', markersize=12, color='#333333', linestyle='None', alpha=0.9, label='Measured data')

if np.any(y_fit):
    ax2.plot(x, y_fit, marker='o', color='#d62728', linestyle='None', alpha=0.5, label='Fitted data')

if np.any(y_fit_arr):
    fit_label = (f'Probabilistic fit:\n$\\tau={tau_fit:.0f}$, $\\eta={eta_fit:.2f}$,\n'
                 f'$Y_{{sat}}={Y_eq_fit:.2f}, Y_{0}={Y0_fit:.2f}$, RMS={fiterror:.2f}')
    ax2.plot(t_arr, y_fit_arr, color='#d62728', linestyle='-.', lw=1.2, alpha=0.8, label=fit_label)

# Adaptation Time Annotations
y_adapt = 0.7819937  
t_adapt = -T0 * np.log((1 - y_adapt) / B_INIT)

ax2.plot([0, 90], [y_adapt, y_adapt], color='blue', linestyle='--', lw=1.5, alpha=0.5)
ax2.plot([t_adapt, t_adapt], [0.2, y_adapt], color='blue', linestyle='--', lw=1.5, alpha=0.5)
ax2.plot(t_adapt, y_adapt, marker='o', color='blue')
ax2.text(t_adapt - 5, 0.16, 'Adaptation\ntime', color='blue', ha='center', va='top')

# Formatting Panel b
ax2.set(xlabel='Acquisition time (s)', ylabel='Available pigment', xlim=(0, FINAL_T), ylim=(0.2, 1.02))
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
plt.savefig('./figs/fig7_protocol_bleach_simu.png', dpi=600, bbox_inches='tight')

plt.show()