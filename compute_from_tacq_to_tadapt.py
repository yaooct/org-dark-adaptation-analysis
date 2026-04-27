# -*- coding: utf-8 -*-
"""
Cone Dark Adaptation Protocol & Simulation
Description: Simulates photopigment regeneration, accounting for fractional 
             bleaching caused by sequential probing flashes, and visualizes 
             the conversion from acquisition time to effective adaptation time.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import plot_configuration_manuscript as pcfg
pcfg.setup()

# ==========================================
# 1. PARAMETERS & SIMULATION
# ==========================================
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


# ==========================================
# 2. FIGURE GENERATION
# ==========================================
# Use a single axis since we are only plotting the simulation
fig, ax = plt.subplots(figsize=(7.2, 3.2))

# ------------------------------------------
# Plot Unperturbed Regeneration Curve
# ------------------------------------------
t_baseline = np.arange(FINAL_T)
avail_curve_ideal = 1 - (B_INIT * np.exp(-t_baseline / T0))

ax.plot(t_baseline, avail_curve_ideal, color='blue', lw=1.5, 
         label=f"Pigment regeneration \n($t_{{0}} = {T0}\\,\\mathrm{{s}}$)")

# ------------------------------------------
# Plot Perturbed Trajectory & Drops
# ------------------------------------------
# Continuous bleaching/regeneration trajectory
ax.plot(all_times, all_avail_pigment, color='black', linestyle='--', lw=1.2,
         label=f"Simulated remaining \npigment level")

# Fractional bleaching drop lines
for i in range(0, len(marker_times), 2):
    lbl = f'Pigment bleached \nby test flash' if i == 0 else None
    ax.plot(marker_times[i:i+2], marker_values_avail[i:i+2], color='black', lw=2.5, label=lbl)
    
# Fractional bleaching 
print(f'effective bleach level:')
for i in range(0, len(marker_times), 2):
    print(f'{(marker_values_avail[i]-marker_values_avail[i+1]):.2f}')

# Pre-flash data points
x_data = TIMES
y_data = marker_values_avail[0::2]
ax.plot(x_data, y_data, marker='*', markersize=8, color='red', 
        linestyle='None', alpha=0.9, label='Measurement')

# ------------------------------------------
# Adaptation Time Conversion
# ------------------------------------------
# Pick a specific acquisition time to highlight (e.g., t_acq = 90s)
t_acq_example = 300
idx = TIMES.index(t_acq_example)
y_adapt = y_data[idx]

ax.plot([t_acq_example, t_acq_example], [0.2, y_adapt], color='black', linestyle=':', lw=1.5, alpha=0.6)
 
# Calculate the corresponding adaptation time
t_adapt = -T0 * np.log((1 - y_adapt) / B_INIT)
print(f'adaptation time:\n{(-T0 * np.log((1 - y_data) / B_INIT)).round()}s')
# Draw mapping lines
ax.plot([0, t_acq_example], [y_adapt, y_adapt], color='blue', linestyle=':', lw=1.5, alpha=0.6)
ax.plot([t_adapt, t_adapt], [0.2, y_adapt], color='blue', linestyle=':', lw=1.5, alpha=0.6)
ax.plot(t_adapt, y_adapt, marker='o', color='blue', markersize=6)

ax.text(t_adapt - 5, 0.22, rf'$t_{{adapt}}$' + '\n' + f'({t_adapt:.1f} s)', 
        color='blue', ha='right', va='bottom')
ax.text(t_acq_example - 5, 0.22, rf'$t_{{acq}}$' + '\n' + f'({t_acq_example} s)', 
        color='black', ha='right', va='bottom')

# %%
# ==========================================
# 2. PROBABILISTIC CURVE FITTING
# ==========================================
# ==========================================
# 2. MATHEMATICAL MODELS
# ==========================================
import scipy.optimize as spo
import matplotlib.pyplot as plt

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


lower_bounds = [30.0, -0.2, 0.0, 0.0] 
upper_bounds = [180.0, 0.2, 1.0, 1.0]
p0 = [60.0, -0.08, 0.02, 0.5]  # [tau, eta, Y_eq, Y0]
bounds = (lower_bounds, upper_bounds)

x = np.array(x_data)
y = np.array(y_data)
t_arr = np.linspace(0, x[-1] + 50, 1000)

res, _ = spo.curve_fit(objective_function, x, y, p0=p0, bounds=bounds, maxfev=50000, ftol=1e-10)
tau_fit, eta_fit, Y_eq_fit, Y_start_fit = res

y_fit = objective_function(x, tau_fit, eta_fit, Y_eq_fit, Y_start_fit)
fiterror = np.sqrt(np.nanmean((y - y_fit) ** 2))

y_fit_arr = yperturb(t_arr, tau_fit, eta_fit, Y_eq_fit, Y_start_fit, x)
Y0_fit = y_fit_arr[0]

# if np.any(y_fit):
#     ax.plot(x, y_fit, marker='o', color='#d62728', linestyle='None', alpha=0.5, label='Fitted data')

if np.any(y_fit_arr):
    fit_label = (f'Probabilistic fit:\n$\\tau={tau_fit:.0f}$, $\\eta={eta_fit:.2f},Y_{{sat}}={Y_eq_fit:.2f}$,\n '
                 f'$Y_{0}={Y0_fit:.2f}$, RMS={fiterror:.2f}')
    ax.plot(t_arr, y_fit_arr, color='#d62728', linestyle='-.', lw=1.2, alpha=1, label=fit_label)


# %%
# ------------------------------------------
# Formatting
# ------------------------------------------
ax.set_xlabel('Time (s)')
ax.set_ylabel('Available Pigment')
ax.set_xlim(0, FINAL_T)
ax.set_ylim(0.2, 1.02)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val * 100:.0f}%'))

ax.legend(frameon=False, loc='best')


# Uncomment to save the figure automatically
os.makedirs('./figs', exist_ok=True)
plt.savefig('./figs/fig_compute_tadapt_2models.png', dpi=600, bbox_inches='tight')

plt.show()