# -*- coding: utf-8 -*-
"""
Cone ORG Dark Adaptation Visualization

Author: YC
Date: April 2026
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load publication aesthetics
import plot_configuration_manuscript as pcfg
pcfg.setup()


def calculate_tadapt(b, t0, b_init=0.67):
    return -t0 * np.log(b / (b_init * 100))


# %% ==== Fig 1: One example to show the 5 mins ORG responses ====
DATA_DIR = r"./data"       # Update to your local data directory
OUTPUT_DIR = r"./figs"     # Update to your local output directory
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 1. Load the file
data_filepath = os.path.join(DATA_DIR, r'ORG_curves_304s_example.npz')
data = np.load(data_filepath, allow_pickle=True)

# 2. Extract your variables
tall_loaded = data['tall']
oplall_loaded = data['oplall']
t_vals_loaded = data['t_vals']

# 3. Create the plot
plt.figure(figsize=(7.2, 2.3))
cmap_colors = ["green", "#FFB2B2", "#FF8080", "#FF4D4D", "#FF1A1A", 
               "#E60000", "#B80000", "#8B0000", "#5C0000", "#2E0000"]

for kk, (t_arr, opl_arr, sec) in enumerate(zip(tall_loaded, oplall_loaded, t_vals_loaded)):
    # Pick a color (use modulo just in case there are more curves than colors)
    c_color = cmap_colors[kk % len(cmap_colors)] 
    plt.plot(t_arr, opl_arr, '-', color=c_color, label=f'{sec}s')

plt.ylabel('ΔOPL (nm)')
plt.xlabel('Acquisition time (s)')
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.grid(True, alpha=0.12)

# Save and show
plt.savefig(os.path.join(OUTPUT_DIR, r'fig1_ORG_curves_304s.png'), dpi=600, bbox_inches='tight')
plt.show()


# %% ==== Fig 2: Example of averaged cone ORG curve for two subjects ====

# 1. File Paths & Setup
data_file_path_1 =  os.path.join(DATA_DIR, r'fitRMS30_sub1_rep2_plotdata.pkl')
data_file_path_2 =  os.path.join(DATA_DIR, r'fitRMS30_sub2_rep7_plotdata.pkl')

cmap = ["#FFB2B2", "#FF8080", "#FF4D4D", "#FF1A1A", 
        "#E60000", "#B80000", "#8B0000", "#5C0000", "#2E0000"]

# Create a 2x2 grid
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7.2, 3.5), 
                        gridspec_kw={'wspace': 0.3, 'hspace': 0.44, 'width_ratios': [2, 1]})


# ==========================================
# Top Row: Subject 1 (Panels a, b)
# ==========================================
with open(data_file_path_1, 'rb') as f:
    data1 = pickle.load(f)

# Panel Labels
axs[0, 0].text(-0.12, 1.1, 'a', transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold', va='top', ha='right')
axs[0, 1].text(-0.22, 1.1, 'b', transform=axs[0, 1].transAxes, fontsize=13, fontweight='bold', va='top', ha='right')

for trial in data1['trials']:
    k = trial['k_index']
    color = cmap[k % len(cmap)] 
    
    # Compute the adaptation time
    pigment_remain = trial['frac_b'] * 100 / 8
    pigment_bleached = 100 - pigment_remain * 100  # %
    Tadaptall = calculate_tadapt(pigment_bleached, 60).astype(float)
    
    label_str = f"{trial['Tacq_val']} - {Tadaptall:.0f}"
    
    # Left Panel (Short time ORG)
    axs[0, 0].plot(trial['torg'], trial['Lmean'], marker='.', color=color, 
                   linestyle='None', markersize=6, alpha=0.2, markeredgewidth=0.5)
    axs[0, 0].plot(trial['torg'], trial['Lfitmean'], linestyle='-', color=color, lw=1.2)

    # Right Panel (Long time model prediction)
    axs[0, 1].plot(trial['t_model'], trial['yfitopl'], linestyle='-', color=color, lw=1.2, label=label_str)

# Format Axes
axs[0, 0].set_ylim(-10, 155)
axs[0, 0].set_xlim(-0.03, 2.03)
axs[0, 0].set_ylabel('ΔOPL (nm)') # Only need X-axis label on the bottom row

axs[0, 1].set_ylim(-10, 155)
axs[0, 1].set_xlim(-0.03, 20)
axs[0, 1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False,
                 title=r"$\mathbf{t_{acq}}$ (s) $-$ $\mathbf{t_{adapt}}$ (s)", 
                 title_fontproperties={'weight': 'bold'})


# ==========================================
# Bottom Row: Subject 2 (Panels c, d)
# ==========================================
with open(data_file_path_2, 'rb') as f:
    data2 = pickle.load(f)

# Panel Labels
axs[1, 0].text(-0.12, 1.1, 'c', transform=axs[1, 0].transAxes, fontsize=13, fontweight='bold', va='top', ha='right')
axs[1, 1].text(-0.22, 1.1, 'd', transform=axs[1, 1].transAxes, fontsize=13, fontweight='bold', va='top', ha='right')

for trial in data2['trials']:
    k = trial['k_index']
    color = cmap[k % len(cmap)] 
    
    # Compute the adaptation time
    pigment_remain = trial['frac_b'] * 100 / 8
    pigment_bleached = 100 - pigment_remain * 100  # %
    Tadaptall = calculate_tadapt(pigment_bleached, 60).astype(float)
    
    label_str = f"{trial['Tacq_val']} - {Tadaptall:.0f}"
    
    # Left Panel (Short time ORG)
    axs[1, 0].plot(trial['torg'], trial['Lmean'], marker='.', color=color, 
                   linestyle='None', markersize=6, alpha=0.2, markeredgewidth=0.5)
    axs[1, 0].plot(trial['torg'], trial['Lfitmean'], linestyle='-', color=color, lw=1.2)

    # Right Panel (Long time model prediction)
    axs[1, 1].plot(trial['t_model'], trial['yfitopl'], linestyle='-', color=color, lw=1.2, label=label_str)

# Format Axes
axs[1, 0].set_ylim(-10, 155)
axs[1, 0].set_xlim(-0.03, 2.03)
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('ΔOPL (nm)')

axs[1, 1].set_ylim(-10, 155)
axs[1, 1].set_xlim(-0.03, 20)
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False,
                 title=r"$\mathbf{t_{acq}}$ (s) $-$ $\mathbf{t_{adapt}}$ (s)", 
                 title_fontproperties={'weight': 'bold'})


# Ensure y-axis labels align cleanly across the columns
fig.align_ylabels(axs[:, 0])

# Save and show
plt.savefig(os.path.join(OUTPUT_DIR, r"fig2_2_exORGtraces_2subs.png"), dpi=600, bbox_inches='tight')
plt.show()