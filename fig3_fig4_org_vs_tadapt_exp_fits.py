# -*- coding: utf-8 -*-
"""
Cone Optoretinography (ORG) Parameter Fitting and Visualization

Description:
    This script processes ORG data to evaluate how amplitude and dynamic 
    parameters ( maximum OPL change, time-to-peak, rate of late contraction) 
    recover during dark adaptation across multiple subjects. It fits the 
    extracted measurements to an exponential recovery model.

Author: YC
Date: April 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Attempt to load local publication aesthetics configuration
try:
    import plot_configuration_manuscript as pcfg
    pcfg.setup()
except ImportError:
    print("Warning: 'plot_configuration_manuscript' not found. Using default matplotlib styles.")
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})


# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
# File paths
DATA_DIR = r"./data"       # Update to your local data directory
OUTPUT_DIR = r"./figs"     # Update to your local output directory

# Analysis Parameters
SUBJECTS = ['sub1', 'sub2', 'sub3']

# Toggle which metrics to plot. 
# Options: 'opl0406avg', 'OPL_pk_fit (nm)', 't_pk', 'tau_b_fit'
METRICS = [ 'opl0406avg', 'OPL_pk_fit (nm)']

# Plotting Aesthetics
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def exp_func(t, A, tau, C):
    """Exponential recovery function for curve fitting."""
    return A * np.exp(-t / tau) + C


def calculate_tadapt(b, t0, b_init=0.67):
    """Calculates adaptation time based on the fraction of bleached photopigment."""
    return -t0 * np.log(b / (b_init * 100))


def cal_frac_bleach_percent(times, t0):
    """
    Iteratively calculates the percentage of fractional bleach 
    over a sequence of bleaching flashes.
    """
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


def process_and_fit_data(file_path, subject, param, t0_rege=60):
    """
    Reads patient data, calculates adaptation times, groups by trial, 
    and fits the global data to an exponential model.

    Returns:
        scatter_groups (list): Tuples of (x_vals, y_vals) for each trial.
        fit_data (tuple): (x_fit, y_fit) array for the continuous fit line.
        fit_label (str): Formatted string of the optimized fit parameters.
    """
    try:
        merged_df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return [], None, None

    param_cols = ['opl0406avg', 'OPL_pk_fit (nm)', 't_pk', 'tau_b_fit']
    merged_df['tacq'] = pd.to_numeric(merged_df['tacq'], errors='coerce').astype('Int64')

    df = merged_df.groupby(['tacq', 'bleaching(%)', 'Tall', 'datalabel'])[param_cols].median().reset_index()

    # Filter for dark adaptation replicates
    da07_files = sorted(df[df['datalabel'].str.contains('DArep')]['datalabel'].unique())
    subset_df = df[df['datalabel'].isin(da07_files)]
    grouped_by_tall = subset_df.groupby('Tall')

    scatter_groups = []
    x_all, y_all = [], []

    # Process each trial block
    for tall_value, group_df in grouped_by_tall:
        tall = np.array(tall_value.strip('[]').split(), dtype=int)
        
        # Calculate pigment dynamics
        fracball_percent = cal_frac_bleach_percent(tall, t0_rege)     
        pigment_remain = fracball_percent / 8
        pigment_bleached = 100 - (pigment_remain * 100) 
        
        t_adapt_all = calculate_tadapt(pigment_bleached, t0_rege).astype(float)
        
        # Sort acquisition times cleanly
        group_df = group_df.sort_values(by='tacq', key=lambda x: x.astype(int)).reset_index(drop=True)
        group_df['Tadaptall'] = t_adapt_all
        
        # Filter NaNs
        valid_idx = (~np.isnan(t_adapt_all)) & (~np.isnan(group_df[param]))
        x_val = t_adapt_all[valid_idx]
        y_val = group_df[param][valid_idx]

        if len(x_val) > 0:
            scatter_groups.append((x_val, y_val))
            x_all.extend(x_val)
            y_all.extend(y_val)

    x_all, y_all = np.array(x_all), np.array(y_all)

    if len(x_all) == 0:
        return scatter_groups, None, None

    # Global Curve Fitting
    initial_guess = [np.max(y_all) - np.min(y_all), 60, np.max(y_all)]
    try:
        popt, _ = curve_fit(exp_func, x_all, y_all, p0=initial_guess, maxfev=50000)
        y_pred = exp_func(x_all, *popt)
        
        # R-squared calculation
        ss_res = np.sum((y_all - y_pred) ** 2)
        ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Continuous line for plotting
        x_fit = np.linspace(min(x_all), max(x_all), 500)
        y_fit = exp_func(x_fit, *popt)
        
        # Formatting legend based on the magnitude of the C parameter
        if popt[2] > 50:
            fit_label = f"A={popt[0]:.0f}, $\\tau$={popt[1]:.0f}, C={popt[2]:.0f}, $R^2$={r_squared:.2f}"
        else:
            fit_label = f"A={popt[0]:.1f}, $\\tau$={popt[1]:.0f}, C={popt[2]:.2f}, $R^2$={r_squared:.2f}"
            
        return scatter_groups, (x_fit, y_fit), fit_label

    except Exception as e:
        print(f"Fit failed for {subject} - {param}: {e}")
        return scatter_groups, None, None


# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Configure 2x3 Subplot Grid
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7.2, 3.5), 
                            gridspec_kw={'wspace': 0.75, 'hspace': 0.6})
    panel_letters = ['a', 'b', 'c', 'd', 'e', 'f']
    Y_MIN = 70

    print(f"Processing metrics: {METRICS}")

    # Plotting Loop
    for row_idx, param in enumerate(METRICS):
        for col_idx, subject in enumerate(SUBJECTS):
            ax = axs[row_idx, col_idx]
            file_path = os.path.join(DATA_DIR, f"{subject}_combined_data.csv")
            
            scatter_groups, fit_data, fit_label = process_and_fit_data(file_path, subject, param)

            # Plot raw data scatters
            if scatter_groups:
                for k, (x_vals, y_vals) in enumerate(scatter_groups):
                    ax.plot(x_vals, y_vals, marker='o', linestyle='None', 
                            color=COLORS[k % len(COLORS)], markersize=3, alpha=0.8)
                
                # Plot exponential fit line
                if fit_data:
                    x_fit, y_fit = fit_data
                    ax.plot(x_fit, y_fit, '-k', lw=1.5, label=fit_label)

            # --- Formatting ---
            ax.set_xlim(15, 290)
            ax.set_xlabel('$t_{adapt}$ (s)')
            
            if 't_pk' in param:
                ax.set_ylabel('$t_{peak}$ (s)')
                ax.set_ylim(0.3, 0.99)
            elif 'tau_b' in param:
                ax.set_ylabel('$\\tau_b$ (s$^{-1}$)')
                ax.set_ylim(-0.05, 0.97)
            elif 'opl0406' in param:
                ax.set_ylim(Y_MIN, 167) 
                ax.set_ylabel(r'$\Delta$OPL$_\mathrm{max}$ (nm)')
            elif 'OPL_pk_fit' in param:
                ax.set_ylim(Y_MIN, 167) 
                ax.set_ylabel(r'$\Delta$OPL$_\mathrm{max}^*$ (nm)')

            # Panel Labeling
            panel_idx = row_idx * 3 + col_idx
            ax.text(-0.48, 0.99, panel_letters[panel_idx], transform=ax.transAxes, 
                    fontweight='bold', va='top', ha='right', fontsize=12.5)

            # Clean Legend beneath the plot
            if fit_label:
                ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.3), 
                          frameon=False, handlelength=1.0, fontsize=8)

    # Clean up axis alignment
    fig.align_ylabels(axs[:, 0])

    # Dynamic File Saving based on chosen metrics
    if 't_pk' in METRICS or 'tau_b_fit' in METRICS:
        save_name = 'fig4_exp_fit_taub_tpk_tadapt_3subs.png'
    else:
        save_name = 'fig3_exp_fit_oplmax_tadapt_3subs.png'
        
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Figure saved successfully to: {save_path}")

    plt.show()