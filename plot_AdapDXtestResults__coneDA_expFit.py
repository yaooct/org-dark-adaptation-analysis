# -*- coding: utf-8 -*-
"""
AdaptDx Rapid Dark Adaptation Analysis
Description: Reads XML data, extracts threshold times and log sensitivities, 
             fits an exponential decay model to the early recovery phase, 
             and plots the results with publication-quality formatting.
"""

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
# 1. PARAMETERS & MODELS
# ==========================================
DIR_PATH = r".\data\rapid_adaptDX"
xml_files = glob.glob(os.path.join(DIR_PATH, '*.xml'))

def decay_model(x, A, tau, C):
    """Exponential decay model for dark adaptation."""
    return A * np.exp(-x / tau) + C


# ==========================================
# 2. DATA EXTRACTION & FIGURE GENERATION
# ==========================================
fig, ax = plt.subplots(figsize=(7.2, 3))

# Generate a colormap to differentiate subjects/trials
colors =['C0','C1','C2']

for idx, filename in enumerate(xml_files):
    dataname = os.path.basename(filename).split('.')[0]
    tree = ET.parse(filename)
    root = tree.getroot()

    # Extract valid threshold data
    data = []
    for threshold in root.findall(".//Threshold"):
        validity = threshold.find("threshold_validity").text.lower() == "true"
        if validity:
            time = float(threshold.find("threshold_time_minutes").text) * 60
            log_sens = float(threshold.find("threshold_value").text)
            data.append((time, log_sens))

    if not data:
        print(f"No valid data in {dataname}, skipping...")
        continue

    x_data = np.array([d[0] for d in data])
    y_data = np.array([d[1] for d in data])

    # Fit using the first 5 data points
    x_data_fit = x_data[:5]
    y_data_fit = y_data[:5]

    # Initial guess: [Amplitude, Time Constant, Offset]
    p0 = [max(y_data_fit) - min(y_data_fit), 100, min(y_data_fit)]

    try:
        popt, _ = curve_fit(decay_model, x_data_fit, y_data_fit, p0=p0)
    except RuntimeError:
        print(f"Optimal parameters not found for {dataname}.")
        continue

    A, tau, C = popt

    # Calculate R^2 for the fitted points
    y_fit_points = decay_model(x_data_fit, *popt)
    ss_res = np.sum((y_data_fit - y_fit_points) ** 2)
    ss_tot = np.sum((y_data_fit - np.mean(y_data_fit)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Generate smooth curve for plotting
    x_fit = np.linspace(min(x_data_fit), max(x_data_fit), 500)
    y_fit = decay_model(x_fit, *popt)

    # ------------------------------------------
    # Format the LaTeX Equation for the Legend
    # Note: Double curly braces {{ }} are used so Python f-strings 
    # don't interfere with LaTeX superscript/subscript syntax.
    # ------------------------------------------
    label_str = rf"Fit: $y(t) = {A:.1f} e^{{-t/{tau:.1f}}} + {C:.1f}\quad (R^2={r_squared:.2f})$"

    # Plot data points and fitting curve
    color = colors[idx]
    ax.plot(x_data, y_data, marker='o', linestyle='None', color=color, markersize=5, alpha=0.8, label=f'{dataname}')
    ax.plot(x_fit, y_fit, linestyle='-', color=color, linewidth=1.5, label=label_str)

    print(f"{dataname} - A={A:.4f}, tau={tau:.4f}, C={C:.4f}, R²={r_squared:.4f}")

# ==========================================
# 3. AXES FORMATTING & FINALIZATION
# ==========================================
ax.set_xlabel('Time (s)')
ax.set_ylabel('Log Sensitivity')
ax.set_xlim(0, 370)

# Remove top and right spines for a clean, publication-ready look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add the legend
ax.legend(frameon=False, ncols = 1, loc='best')
save_path = r'.\figs\AdaptDX_testResutls_3subs.png'
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()