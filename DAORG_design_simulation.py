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

# ==========================================
# 1. PARAMETERS & SIMULATION
# ==========================================
T0 = 60.0             # Time constant (s)
B_INIT = 0.65         # Initial bleaching level
B_OFFSET = 0.21       # Fractional bleach from probing flashes
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

# Calculations for export
x_data = TIMES
y_data = marker_values_avail[0::2]  # Pre-flash
y_after = marker_values_avail[1::2] # Post-flash

effective_bleach = y_data - y_after
t_adapt_array = -T0 * np.log((1 - y_data) / B_INIT)

# ==========================================
# 2. DATA EXPORT TO .TXT
# ==========================================
os.makedirs('./output', exist_ok=True)
txt_path = './output/experiment_design.txt'

with open(txt_path, 'w') as f:
    f.write("Cone Dark Adaptation Protocol - Experiment Design\n")
    f.write("=================================================\n\n")
    f.write("[ Parameters ]\n")
    f.write(f"T0 (Time constant)     : {T0} s\n")
    f.write(f"B_INIT (Initial bleach): {B_INIT}\n")
    f.write(f"B_OFFSET (probe bleach): {B_OFFSET}\n")
    f.write(f"Acq. Time (deliver test flashes): {TIMES}\n\n")
    
    f.write("[ Simulation Results ]\n")
    f.write(f"{'Acq. Time (s)':<18} | {'Effective Bleach':<20} | {'Adaptation Time (s)':<20}\n")
    f.write("-" * 65 + "\n")
    for i, t in enumerate(TIMES):
        f.write(f"{t:<18} | {effective_bleach[i]:<20.2f} | {t_adapt_array[i]:<20.0f}\n")

print(f"Data successfully exported to {txt_path}")

# ==========================================
# 3. FIGURE GENERATION
# ==========================================
fig, ax = plt.subplots(figsize=(7.2, 2))

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
ax.plot(all_times, all_avail_pigment, color='black', linestyle='--', lw=1.2,
        label=f"Simulated remaining \npigment level")

# Fractional bleaching drop lines
for i in range(0, len(marker_times), 2):
    lbl = f'Pigment bleached \nby test flash' if i == 0 else None
    ax.plot(marker_times[i:i+2], marker_values_avail[i:i+2], color='black', lw=2.5, label=lbl)

# Pre-flash data points
ax.plot(x_data, y_data, marker='*', markersize=12, color='red', 
        linestyle='None', alpha=0.9, label='Measurement')

# ------------------------------------------
# Adaptation Time Conversion Visualization
# ------------------------------------------
t_acq_example = 300
idx = TIMES.index(t_acq_example)
y_adapt = y_data[idx]
t_adapt = t_adapt_array[idx]

ax.plot([t_acq_example, t_acq_example], [0.2, y_adapt], color='black', linestyle=':', lw=1.5, alpha=0.6)

# Draw mapping lines
ax.plot([0, t_acq_example], [y_adapt, y_adapt], color='blue', linestyle=':', lw=1.5, alpha=0.6)
ax.plot([t_adapt, t_adapt], [0.2, y_adapt], color='blue', linestyle=':', lw=1.5, alpha=0.6)
ax.plot(t_adapt, y_adapt, marker='o', color='blue', markersize=6)

ax.text(t_adapt - 5, 0.22, rf'$t_{{adapt}}$' + '\n' + f'({t_adapt:.1f} s)', 
        color='blue', ha='right', va='bottom', fontsize=9)
ax.text(t_acq_example - 5, 0.22, rf'$t_{{acq}}$' + '\n' + f'({t_acq_example} s)', 
        color='black', ha='right', va='bottom', fontsize=9)

# ------------------------------------------
# Formatting
# ------------------------------------------
ax.set_xlabel('Time (s)')
ax.set_ylabel('Available Pigment')
ax.set_xlim(0, FINAL_T)
ax.set_ylim(0.2, 1.02)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val * 100:.0f}%'))
ax.legend(frameon=False, loc='best')

# Save the figure automatically
os.makedirs('./figs', exist_ok=True)
plt.savefig('./figs/fig_compute_tadapt.png', dpi=600, bbox_inches='tight')

plt.show()