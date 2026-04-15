# -*- coding: utf-8 -*-
"""
Cone ORG Rate-Limited Model Fitting & Pigment Simulation

Description:
    Generates a 3x3 figure plotting the recovery of ORG metrics 
    (ΔOPL_max, t_peak, tau_b) against adaptation time across three subjects. 
    This script contains its own internal engine for simulating fractional 
    photopigment bleaching and regeneration to calculate exact adaptation times.

Author: YC
Date: April 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
from scipy.optimize import curve_fit
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
# 1. GLOBAL CONFIGURATION & PARAMETERS
# ==========================================
# Paths
DATA_DIR = r"./data"          # Replace with your actual data directory path
OUTPUT_DIR = r"./figs"        # Replace with your output path

SUBJECTS = ['sub1', 'sub2', 'sub3']
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# --- Simulation Parameters ---
KM_SIM = 0.2
V_SIM = 0.5 / 60          # s^-1
B_INIT = 0.67             # initial bleach fraction
T0_SIM = 60               # exponential time constant
B_OFFSET = 0.08           # fractional bleach multiplier
FINAL_T = 500

# --- Metrics Configuration ---
METRICS_CONFIG = [
    {
        'param': 'opl0406avg',
        'ylabel': r'$\Delta$OPL$_\mathrm{max}$ (nm)',
        'ylim': (70, 167),
        'is_tau': False,
        'fmt_y0': '.1f' 
    },
    {
        'param': 't_pk',
        'ylabel': r'$t_{peak}$ (s)',
        'ylim': (0.3, 0.95),
        'is_tau': False,
        'fmt_y0': '.2f'
    },
    {
        'param': 'tau_b_fit',
        'ylabel': r'$\tau_b$ (s$^{-1}$)',
        'ylim': (-0.02, 0.89),
        'is_tau': True,
        'fmt_y0': '.2f'
    }
]


# ==========================================
# 2. INTERNAL SIMULATION ENGINE
# ==========================================
def p_RL(t, b0=B_INIT):
    """Rate-limited regeneration model (fractional pigment)."""
    beta = (1 + KM_SIM) * V_SIM / KM_SIM
    A = (b0 / KM_SIM) * np.exp(b0 / KM_SIM)
    arg = A * np.exp(-beta * t)
    return 1 - KM_SIM * lambertw(arg).real

def RL_inverse_t(p_target, b0=B_INIT):
    """Numerically invert RL model to get t such that p_RL(t, b0) = p_target"""
    def f(t): return p_RL(t, b0) - p_target
    return spo.root_scalar(f, bracket=[0, 1000], method='bisect').root

def p_exp(t, b0=B_INIT, t0=T0_SIM):
    """Exponential regeneration model (fractional pigment)."""
    return 1 - b0 * np.exp(-t / t0)

def exp_inverse_t(p, b0=B_INIT, t0=T0_SIM):
    """Invert exponential model."""
    return -t0 * np.log((1 - p) / b0)

def simulate_rege_bleach(model_forward, model_inverse, times, dt_step=0.1, verbose=False):
    """Simulates iterative fractional bleaching across an acquisition sequence."""
    T, P = [0], [1 - B_INIT]
    t_prev = 0
    b_t = B_INIT
    flash_markers = []

    for t_flash in times:
        # Regeneration between flashes
        t_points = np.arange(t_prev + dt_step, t_flash + dt_step, dt_step)
        t_points = t_points[t_points <= t_flash]
        for t_i in t_points:
            T.append(t_i)
            P.append(model_forward(t_i - t_prev, b_t))

        # Pigment just before flash
        p_before = model_forward(t_flash - t_prev, b_t)
        frac_bleach = B_OFFSET * p_before
        p_after = p_before - frac_bleach

        # Equivalent adaptation time
        t_adapt = model_inverse(p_before, B_INIT)
        
        if verbose:
            print(f"{t_flash}s: p {p_before:.4f}-->{p_after:.4f}, frac_bleach = {frac_bleach:.3f}, t_adapt = {t_adapt:.1f}")
            print("---------------------------")

        flash_markers.append((t_flash, p_before, p_after, t_adapt))

        # Update for next interval
        b_t = 1 - p_after
        t_prev = t_flash
        T.append(t_flash)
        P.append(p_after)

    # Final tail regeneration
    t_points = np.arange(t_prev + dt_step, FINAL_T + dt_step, dt_step)
    t_points = t_points[t_points <= FINAL_T]
    for t_i in t_points:
        T.append(t_i)
        P.append(model_forward(t_i - t_prev, b_t))

    return np.array(T), np.array(P), flash_markers

def run_pigment_simu(times, verbose=False):
    """Executes both RL and Exp simulations and returns extracted adaptation metrics."""
    if verbose: print("=== RL model ==")
    _, _, flashes_RL = simulate_rege_bleach(p_RL, RL_inverse_t, times, verbose=verbose)
    tadapt_RL = [item[-1] for item in flashes_RL]
    fracb_RL = [item[1] - item[2] for item in flashes_RL]
    
    if verbose: print("\n=== Exponential model ===")
    _, _, flashes_exp = simulate_rege_bleach(p_exp, exp_inverse_t, times, verbose=verbose)
    tadapt_exp = [item[-1] for item in flashes_exp]
    fracb_exp = [item[1] - item[2] for item in flashes_exp]
    
    return tadapt_RL, fracb_RL, tadapt_exp, fracb_exp


# ==========================================
# 3. CURVE FITTING TARGET FUNCTION
# ==========================================
def y_of_t_RL(t, Km, v, B0, y_da):
    """Target function used by curve_fit to optimize RL parameters against raw data."""
    Km = np.maximum(Km, 1e-10)     
    A_rl = (B0 / Km) * np.exp(B0 / Km)
    beta = (1 + Km) * v / Km
    arg = A_rl * np.exp(-beta * t)    
    p_t = 1 - Km * lambertw(arg).real
    return y_da * p_t


# ==========================================
# 4. MAIN EXECUTION BLOCK (3x3 Plot)
# ==========================================
if __name__ == "__main__":
    
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(7.2, 7.5), gridspec_kw={'wspace': 0.65, 'hspace': 0.7})
    panel_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    for row_idx, config in enumerate(METRICS_CONFIG):
        param = config['param']
        
        for col_idx, subject in enumerate(SUBJECTS):
            ax = axs[row_idx, col_idx]
            file_path = os.path.join(DATA_DIR, f"{subject}_combined_data.csv")
            
            try:
                merged_df = pd.read_csv(file_path, low_memory=False)
            except FileNotFoundError:
                print(f"Skipping {subject}: File not found at {file_path}")
                continue

            # Data Prep
            merged_df['tacq'] = pd.to_numeric(merged_df['tacq'], errors='coerce').astype('Int64')
            df = merged_df.groupby(['tacq', 'bleaching(%)', 'Tall', 'datalabel'])[[param]].median().reset_index()
            
            # Filter for DArep runs
            da07_files = sorted(df[df['datalabel'].str.contains('DArep')]['datalabel'].unique())
            subset_df = df[df['datalabel'].isin(da07_files)]
            grouped_by_tall = subset_df.groupby('Tall')

            x_all, y_all = [], []
            scatter_groups = []
            
            # --- Data Extraction per Trial Group ---
            for k, (tall_value, group_df) in enumerate(grouped_by_tall):
                tall = np.array(tall_value.strip('[]').split(), dtype=float)
                
                # compute the fractional bleach levels and adaptation time
                tadapt_RL, fracb_RL, tadapt_exp, fracb_exp = run_pigment_simu(tall, verbose=True)
                Tadaptall = np.array(tadapt_RL, dtype=float)
                
                group_df = group_df.sort_values(by='tacq', key=lambda x: x.astype(float)).reset_index(drop=True)
                y_vals = np.array(group_df[param], dtype=float)
                
                # Drop NaNs
                valid_idx = (~np.isnan(Tadaptall)) & (~np.isnan(y_vals)) & (Tadaptall > 0) & (y_vals > 0)
                x_valid = Tadaptall[valid_idx]
                y_valid = y_vals[valid_idx]
                
                if len(x_valid) > 0:
                    scatter_groups.append((x_valid, y_valid, COLORS[k % len(COLORS)]))
                    x_all.extend(x_valid)
                    y_all.extend(y_valid)

            x_all, y_all = np.array(x_all), np.array(y_all)

            # --- Plotting Scatter Data ---
            for (x_vals, y_vals, color) in scatter_groups:
                ax.plot(x_vals, y_vals, marker='o', linestyle='None', color=color, markersize=2.5, alpha=0.85)

            # --- Curve Fitting ---
            if len(x_all) > 3:
                y_max = np.max(y_all)
                if config['is_tau']:
                    initial_guess_RL = [0.2, 0.4/60, 0.7, 0.9] # Km, v, b0, y0
                    bounds_RL = ([0.199, 0.3/60, 0.1, 0.1], [0.2, 1/60, 1.0, 1.0])
                    y_all = 1 - y_all  # Invert data for tau fitting
                else:
                    initial_guess_RL = [0.2, 0.4/60, 0.7, y_max]
                    bounds_RL = ([0.199, 0.2/60, 0.1, y_max/100], [0.2, 1/60, 1.0, y_max*100])

                try:
                    popt_RL, _ = curve_fit(y_of_t_RL, x_all, y_all, p0=initial_guess_RL, bounds=bounds_RL, maxfev=1000000)
                    Km_fit, v_fit, b0_fit, y0_fit = popt_RL
                    
                    # Calculate R-squared
                    y_pred = y_of_t_RL(x_all, *popt_RL)
                    ss_res = np.sum((y_all - y_pred) ** 2)
                    ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    # Generate Fit Line
                    x_fit = np.linspace(np.min(x_all), np.max(x_all), 500)
                    y_fit = y_of_t_RL(x_fit, *popt_RL)
                    
                    fmt_y0 = config['fmt_y0']
                    label_RL = (f'$v={v_fit*60:.2f}$/min, $y_0={y0_fit:{fmt_y0}}$,\n'
                                f'$b_0={b0_fit:.2f}$, $R^2={r_squared:.2f}$')
                    
                    if config['is_tau']:
                        ax.plot(x_fit, 1 - y_fit, '-k', lw=1.5, label=label_RL)
                    else:
                        ax.plot(x_fit, y_fit, '-k', lw=1.5, label=label_RL)
                    
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), frameon=False, handlelength=1)
                    
                except Exception as e:
                    print(f"Fit failed for {subject} - {param}: {e}")

            # --- Panel Formatting ---
            ax.set_xlim(15, 290)
            ax.set_ylim(config['ylim'])
            ax.set_xlabel('$t_{adapt}$ (s)')
            ax.set_ylabel(config['ylabel']) 

            ax.grid(True, color='0.85', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            panel_idx = row_idx * 3 + col_idx
            ax.text(-0.35, 1.05, panel_letters[panel_idx], transform=ax.transAxes, 
                    fontweight='bold', va='top', ha='right', fontsize=13)

    fig.align_ylabels(axs[:, 0])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = "fig8_RLfit_oplmax_taub_tpk_taadpt_3subs.png"
    plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=600, bbox_inches='tight')

    plt.show()