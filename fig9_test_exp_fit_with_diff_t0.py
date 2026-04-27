# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:06:49 2026

@author: ycai
"""

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# ==========================================
# 1. PUBLICATION AESTHETICS (Nature Style)
# ==========================================
import plot_configuration_manuscript as pcfg
pcfg.setup()



# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_tadapt(b, t0, b_init=0.67):
    return -t0 * np.log(b / (b_init * 100))

def exp_func(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def cal_frac_bleach_percent(times, t0):
    b_init, b_offset = 0.67, 0.08
    b_prev, t_prev = b_init, 0
    frac_b_all = []
    for t in times:
        b = b_prev * np.exp(-(t - t_prev) / t0)
        frac_b = b_offset * (1 - b)
        b_prev = b + frac_b
        t_prev = t
        frac_b_all.append(frac_b)
    return np.array(frac_b_all) * 100

def fit_for_t0_rege(t0_rege, grouped_by_tall, subject, param):
    """
    Run the full analysis pipeline for a single t0_rege.
    Returns fitted tau_fit for the combined data.
    """
    tlist, ylist = [], []

    for tall_value, group_df in grouped_by_tall:   
        tall = np.array(tall_value.strip('[]').split(), dtype=float)
        fracball_percent = cal_frac_bleach_percent(tall, t0_rege)     
        pigment_remain = fracball_percent / 8
        pigment_bleached = 100 - pigment_remain * 100  # %
        
        # Ensure Tadaptall is float to accept np.nan
        Tadaptall = calculate_tadapt(pigment_bleached, t0_rege).astype(float)
        
        # Sort group_df by 'tacq'
        group_df = group_df.sort_values(by='tacq', key=lambda x: x.astype(float), ascending=True).reset_index(drop=True)
        
            
        group_df['Tadaptall'] = Tadaptall
        
        tlist.append(Tadaptall)
        ylist.append(group_df[param])            

    # --- global fit ---
    x = np.concatenate(tlist)
    y = np.concatenate(ylist)
    
    # Filter valid pairs
    valid = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
    x = x[valid]
    y = y[valid]

    if len(x) < 3 or len(y) < 3:
        return np.nan

    initial_guess = [np.max(y) - np.min(y), 60, np.max(y)]
    
    try:
        popt, pcov = curve_fit(exp_func, x, y, p0=initial_guess, maxfev=50000)
        return popt[1] # Return tau_fit
    except Exception:
        return np.nan

# %%
# ==========================================
# 3. PRE-LOAD & FILTER DATA (Optimization)
# ==========================================

dirfig = r'C:\Users\ycai\OneDrive\0.CHOIR_YC\5.Paper\DA_ORG_paper\figs0409'
dirfig2 = r'./figs'


folder_path = r"./data"
subjects = ['sub1', 'sub2', 'sub3']
param_cols = ['opl0406avg', 'OPL_pk_fit (nm)', 't_pk', 'tau_b_fit']

# Dictionary to hold pre-filtered dataframes to save I/O time
subject_data = {}

for subject in subjects:
    file_path = os.path.join(folder_path, f'{subject}_combined_data.csv')
    try:
        merged_df = pd.read_csv(file_path, low_memory=False)
        merged_df['tacq'] = pd.to_numeric(merged_df['tacq'], errors='coerce').astype('Int64')
        df = merged_df.groupby(['tacq','bleaching(%)', 'Tall','datalabel'])[param_cols].median().reset_index()
        
        da07_files = sorted(df[df['datalabel'].str.contains('DArep') & ~df['datalabel'].str.contains('09DArep')]['datalabel'].unique())
        subset_df = df[df['datalabel'].isin(da07_files)]
        
        subject_data[subject] = subset_df.groupby('Tall')
    except FileNotFoundError:
        print(f"Skipping {subject}: CSV not found.")

# ==========================================
# 4. MAIN PLOTTING LOOP
# ==========================================
t0_values = np.linspace(35, 200, 200)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
panel_letters = ['a', 'b', 'c', 'd']
ylabels = [
    r'$\tau$ for $\Delta$OPL$_\mathrm{max}$ (s)',
    r'$\tau$ for $\Delta$OPL$_\mathrm{max}^*$ (s)',
    r'$\tau$ for $t_{peak}$ (s)',
    r'$\tau$ for $\tau_b$ (s)'
]

# Create 2x2 Grid
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(7.2, 1.8), gridspec_kw={'wspace': 0.8, 'hspace': 0.3})
axs = axs.flatten()

for iparam, param in enumerate(param_cols):
    ax = axs[iparam]
    
    for isub, subject in enumerate(subjects):
        if subject not in subject_data:
            continue
            
        grouped_by_tall = subject_data[subject]
        tau_values = []
        
        # Sweep over t0_rege values
        for t0 in t0_values:
            tau = fit_for_t0_rege(t0, grouped_by_tall, subject, param)
            tau_values.append(tau)
        
        tau_values = np.array(tau_values, dtype=float)
        
        # Plot Line
        ax.plot(t0_values, tau_values, '-', color=colors[isub], lw=1.5, label=f'Subject {isub+1}')
    
    # Formatting the subplot
    ax.set_xlabel(r"$t_{0}$ (s)")
    ax.set_ylabel(ylabels[iparam])
    ax.set_xlim(32, 199)
    ax.set_ylim(15,119)
    
    # Nature Styles
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.3)
    
    # Panel Label (Top Left, Bold)
    ax.text(-0.35, 1.1, panel_letters[iparam], transform=ax.transAxes, 
            fontweight='bold', va='top', ha='right', fontsize=12)
    
    # Add Legend only to the first panel
    if iparam == 0:
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(3, -0.3), ncol=3 )

plt.tight_layout()

# Save command (optional)
plt.savefig(os.path.join(dirfig, "supp1_tau_ORGmetrics__for_diff_t0.png"), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(dirfig2, "supp1_tau_ORGmetrics__for_diff_t0.png"), dpi=600, bbox_inches='tight')

plt.show()