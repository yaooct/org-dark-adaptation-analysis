# -*- coding: utf-8 -*-
"""
Probabilistic Model Fitting for Cone ORG Recovery (Single & Global)

Description:
    This script evaluates cone optoretinography (ORG) measurements during 
    dark adaptation. 
    - Phase 1 generates a 1x2 example fit for a single trial.
    - Phase 2 generates a 3x3 global minimization fit across multiple 
      subjects and trials.
      
    It includes a self-contained mathematical engine for exponential 
    systems with stepwise multiplicative perturbations.

Author: YC
Date: April 2026
"""

import os
import numpy as np
import scipy.optimize as spo
import pandas as pd
import matplotlib.pyplot as plt

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
# 1. GLOBAL CONFIGURATION
# ==========================================
DATA_DIR = r"./data"       # Update this to your data directory
OUTPUT_DIR = r"./figs"     # Update this to your output directory
SUBJECTS = ['sub1', 'sub2', 'sub3']

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared Color Palette (tab10)
colors_map = plt.cm.get_cmap('tab10')
COLORS = [colors_map(i % 10) for i in range(10)]


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


def overall_rss(P, global_data):
    """
    Calculates the total Residual Sum of Squares (RSS) for all trials 
    during global minimization.
    P: [tau, eta, Y_eq, Y0]
    """
    tau, eta, Y_eq, Y0 = P[0], P[1], P[2], P[3]
    
    if tau <= 0: return np.inf
    lamb = 1.0 / tau
    total_rss = 0.0
    
    for k, data in enumerate(global_data):
        t_fit = data['t_fit']
        opl_fit = data['opl_fit']
        T_start = data['T_start']
        
        # Calculate required starting value based on exponential growth
        Y_start_k = Y_eq - (Y_eq - Y0) * np.exp(-lamb * T_start)
        
        try:
            y_pred = objective_function_helper(tau, eta, Y_eq, Y_start_k, t_fit)
            if y_pred.shape != opl_fit.shape: return np.inf

            residuals = opl_fit - y_pred
            valid_residuals = residuals[~np.isnan(residuals)] 
            total_rss += np.sum(valid_residuals**2)
            
        except Exception:
            return np.inf 
            
    return total_rss


# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":

    # ---------------------------------------------------------
    # PHASE 1: SINGLE TRIAL EXAMPLE (1x2 Plot)
    # ---------------------------------------------------------
    print("Generating Phase 1: Single Trial Example...")

    DATA_FILE_EX = os.path.join(DATA_DIR, "sub1_median_data.csv")
    TARGET_TRIAL = 'DArep2'

    try:
        df_ex = pd.read_csv(DATA_FILE_EX, low_memory=False)
        df_ex['tacq'] = pd.to_numeric(df_ex['tacq'], errors='coerce').astype('Int64')
        
        subset_df_ex = df_ex[df_ex['datalabel'] == TARGET_TRIAL]
        datalabels_ex = np.unique(subset_df_ex['datalabel'])

        fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7.2, 2), gridspec_kw={'wspace': 0.3})
        
        configs_ex = [
            {'ax': ax1, 'param': 'opl0406avg', 'ylabel': r'$\Delta$OPL$_\mathrm{max}$ (nm)', 'ylim': (60, 165),
             'guess': [98.0, 0.08, 140.0], 'bounds': ([95.0, 0.0, 50.0, 20.0], [180.0, 0.5, 300.0, 200.0]), 'fmt': '.0f'},
            {'ax': ax2, 'param': 'tau_b_fit', 'ylabel': r'$\tau_b$ (s$^{-1}$)', 'ylim': (-0.02, 0.98),
             'guess': [87.0, -0.1, 0.5], 'bounds': ([20.0, -0.2, 0.0, 0.0], [92.0, 0.0, 1.0, 0.9]), 'fmt': '.2f'}
        ]

        for idx, cfg in enumerate(configs_ex):
            ax = cfg['ax']
            param = cfg['param']
            bounds = cfg['bounds']
            tau_guess, eta_guess, Y_eq_guess = cfg['guess']

            ax.text(-0.15, 1.05, ['a', 'b'][idx], transform=ax.transAxes, fontweight='bold', va='top', ha='right', fontsize=12.5)

            for ik, datarep in enumerate(datalabels_ex):
                dfrep1 = subset_df_ex[subset_df_ex['datalabel'] == datarep]
                
                t_fit = np.array(dfrep1['tacq'], dtype=float)
                opl_fit = np.array(dfrep1[param], dtype=float)
                
                valid_mask = ~np.isnan(t_fit) & ~np.isnan(opl_fit)
                t_fit, opl_fit = t_fit[valid_mask], opl_fit[valid_mask]
                
                if len(t_fit) > 6:
                    t_arr = np.linspace(0, t_fit[-1] + 50, 3000)
                    p0 = [tau_guess, eta_guess, Y_eq_guess, opl_fit[0]]
                    
                    try:
                        res, _ = spo.curve_fit(objective_function, t_fit, opl_fit, p0=p0, bounds=bounds)
                        y_fit = objective_function(t_fit, *res)
                        fiterror = np.sqrt(np.nanmean((opl_fit - y_fit) ** 2))
                    except Exception:
                        res = p0
                        y_fit, fiterror = np.zeros_like(t_fit), np.nan
                    
                    y_fit_arr = yperturb(t_arr, res[0], res[1], res[2], res[3], t_fit)
                    
                    fmt = cfg['fmt']
                    fit_label = (f'Fit: $\\tau={res[0]:.0f}$, $\\eta={res[1]:.2f}$,\n'
                                 f'$Y_{{sat}}={res[2]:{fmt}}$, $Y_{0}={y_fit_arr[0]:{fmt}}$,\n'
                                 f'RMS={fiterror:.2f}')
                    
                    ax.plot(t_arr, y_fit_arr, '-k', alpha=0.6, lw=1.5, label=fit_label)
                    ax.plot(t_fit, opl_fit, 'r*', alpha=1-ik*0.1, markersize=8, label='Measured data')
                    ax.plot(t_fit, y_fit, 'bo', alpha=1-ik*0.1, markersize=4, label='Fitted data')

            ax.set(xlabel='$t_{acq}$ (s)', ylabel=cfg['ylabel'], xlim=(-10, 505), ylim=cfg['ylim'])
            ax.legend(frameon=False, loc='best')

        plt.savefig(os.path.join(OUTPUT_DIR, "fig5_probfit_ex.png"), dpi=600, bbox_inches='tight')
        plt.show()

    except FileNotFoundError:
        print(f"Skipping Phase 1: {DATA_FILE_EX} not found.")


    # ---------------------------------------------------------
    # PHASE 2: GLOBAL MINIMIZATION (3x3 Plot)
    # ---------------------------------------------------------
    print("Generating Phase 2: Global Minimization Matrix...")

    metrics_config = [
        {'param': 'opl0406avg', 'ylabel': r'$\Delta$OPL$_\mathrm{max}$ (nm)', 'ylim': (60, 165),
         'bounds': ([30, 0.01, 100.0, 0.0], [200.0, 0.2, 500.0, 200.0]), 'guess': [90.0, 0.05, 150.0], 'fmt': '.0f'},
        {'param': 't_pk', 'ylabel': r'$t_{peak}$ (s)', 'ylim': (0.32, 0.98),
         'bounds': ([10.0, 0.02, 0.01, 0.0], [150.0, 0.2, 2.0, 2.0]), 'guess': [30.0, 0.1, 0.6], 'fmt': '.2f'},
        {'param': 'tau_b_fit', 'ylabel': r'$\tau_b$ (s$^{-1}$)', 'ylim': (-0.02, 0.98),
         'bounds': ([10.0, -0.4, 0.0, 0.0], [200.0, 0.2, 1.0, 5.0]), 'guess': [30.0, 0.0, 0.2], 'fmt': '.2f'}
    ]

    fig2, axs2 = plt.subplots(nrows=3, ncols=3, figsize=(7.2, 8), gridspec_kw={'wspace': 0.45, 'hspace': 0.7})
    panel_letters_2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    for row_idx, config in enumerate(metrics_config):
        param = config['param']
        bounds = config['bounds']
        
        for col_idx, subject in enumerate(SUBJECTS):
            ax = axs2[row_idx, col_idx]
            csvfile = os.path.join(DATA_DIR, f"{subject}_median_data.csv")
            
            try:
                df_glob = pd.read_csv(csvfile, low_memory=False)
            except FileNotFoundError:
                continue
                
            df_glob['tacq'] = pd.to_numeric(df_glob['tacq'], errors='coerce').astype('Int64')
            
            # Filter Logic
            da07_files = sorted(df_glob[df_glob['datalabel'].str.contains('DArep')]['datalabel'].unique())        
            subset_df_glob = df_glob[df_glob['datalabel'].isin(da07_files)]
            datalabels_glob = np.unique(subset_df_glob['datalabel'])
            
            global_data = []
            for datarep in datalabels_glob:
                dfrep1 = subset_df_glob[subset_df_glob['datalabel'] == datarep]
                t_fit = np.array(dfrep1['tacq'], dtype=float)
                opl_fit = np.array(dfrep1[param], dtype=float)
                
                valid_idx = (~np.isnan(t_fit)) & (~np.isnan(opl_fit))
                t_fit, opl_fit = t_fit[valid_idx], opl_fit[valid_idx]
                
                if len(t_fit) > 5:
                    global_data.append({'t_fit': t_fit, 'opl_fit': opl_fit, 'T_start': t_fit[0]})
            
            if not global_data:
                continue
                
            # Global Fit Execution
            tau_guess, eta_guess, Y_eq_guess = config['guess']
            p0_global = [tau_guess, eta_guess, Y_eq_guess, np.mean([d['opl_fit'][0] for d in global_data])]
            
            try:
                opt_result = spo.minimize(
                    fun=overall_rss, x0=p0_global, args=(global_data,), 
                    method='L-BFGS-B', bounds=list(zip(bounds[0], bounds[1])),
                    options={'maxiter': 50000, 'ftol': 1e-10}
                )
                tau_opt, eta_opt, Y_eq_opt, Y0 = opt_result.x
                
                # Global R^2
                all_y_data = np.concatenate([d['opl_fit'] for d in global_data])        
                global_r_squared = 1 - (opt_result.fun / np.sum((all_y_data - np.mean(all_y_data))**2))
            except Exception:
                tau_opt, eta_opt, Y_eq_opt, Y0 = p0_global
                global_r_squared = np.nan

            # Plotting the Fit
            t_arr = np.linspace(0, 500, 1000)
            fmt = config['fmt']
            
            for k, data in enumerate(global_data):
                lamb_opt = 1.0 / tau_opt
                Y_start_k = Y_eq_opt - (Y_eq_opt - Y0) * np.exp(-lamb_opt * data['T_start'])
                y_fit_arr = yperturb(t_arr, tau_opt, eta_opt, Y_eq_opt, Y_start_k, data['t_fit'])
                
                if k == 0:
                    text_metric = (f'$\\tau={tau_opt:.0f}$, $\\eta={eta_opt:.2f}$,\n'
                                   f'$Y_{{sat}}={Y_eq_opt:{fmt}}$, $Y_0={Y0:{fmt}}$,\n'
                                   f'$R^2={global_r_squared:.2f}$')
                    ax.plot(t_arr, y_fit_arr, '-k', linewidth=0.8, alpha=0.65, label=text_metric)
                else:
                    ax.plot(t_arr, y_fit_arr, '-k', linewidth=0.8, alpha=0.65)
                
                c = COLORS[k % len(COLORS)]
                ax.plot(data['t_fit'], data['opl_fit'], marker='o', linestyle='None', 
                        markerfacecolor='none', markeredgecolor=c, markersize=4, markeredgewidth=1.2)
            
            # Panel Formatting
            ax.set(xlim=(-10, 505), ylim=config['ylim'], xlabel='$t_{acq}$ (s)')
            if col_idx == 0: ax.set_ylabel(config['ylabel'])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            panel_idx = row_idx * 3 + col_idx
            ax.text(-0.25, 1.05, panel_letters_2[panel_idx], transform=ax.transAxes, 
                    fontweight='bold', va='top', ha='right', fontsize=13)
            ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.2), frameon=False, handlelength=1.0)

    fig2.align_ylabels(axs2[:, 0])

    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_probfit_oplmax_taub_tpk_tacq_3subs.png"), dpi=600, bbox_inches='tight')
    plt.show()

    print("Execution Complete.")