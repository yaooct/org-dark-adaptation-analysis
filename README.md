# Analysis of ORG parameters during dark adaptation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx) This repository contains the custom Python analysis scripts and processed datasets accompanying the manuscript: **"Dark adaptation of the cone optoretinogram"**.

The codebase provides mathematical models to quantify the cone photoreceptors ORG responses during dark adaptation.

## Overview of mathematical models
This repository implements three complementary analytic models to quantify the recovery of ORG amplitude ($\Delta OPL_{max}$) and temporal dynamics (time-to-peak $t_{pk}$, late contraction rate $\tau_{b}$) during dark adaptation:

### 1. Exponential model based on first-order pigment regeneration dynamics (Method 1)
This model evaluates ORG metrics against an effective dark adaptation time ($t_{adapt}$) derived from theoretical exponential photopigment regeneration. The fraction of bleached pigment $b(t)$ during regeneration is modeled as:

$$b(t)=b_{0}e^{-t/t_{0}}$$

where $b_{0}$ is the initial bleach level and $t_{0}$ is the pigment regeneration time constant (assumed to be 60 s). For each ORG measurement, the available pigment fraction $p(t)=1-b(t)$ is calculated, and the ORG metrics are fitted to an exponential recovery function:

$$y(t)=A\cdot e^{-t_{adapt}/\tau}+C$$

where $\tau$ is the recovery time constant and $C$ is the asymptotic saturation level.

### 2.  Rate-limited model of pigment pigment regeneration (Method 2)
This model fits ORG recovery assuming that the process is rate-limited by the delivery of 11-cis-retinoid. The fraction of unbleached pigment $p(t)$ is calculated using the Lambert-W function ($W$):

$$p(t)=1-K_{m}W\left(\frac{b_{0}}{K_{m}}e^{b_{0}/K_{m}}e^{-\frac{1+K_{m}}{K_{m}}vt}\right)$$

where $K_{m}$ is the Michaelis constant (held at 0.2) and $v$ is the initial rate of recovery. ORG metrics are then expressed as a function of the RL-derived adaptation time and fit using a rate-limited scaling relation:

$$y(t_{adapt})=y_{0}\left[1-K_{m}W\left(\frac{b_{0}}{K_{m}}e^{b_{0}/K_{m}}e^{-\frac{1+K_{m}}{K_{m}}vt_{adapt}}\right)\right]$$

### 3. Perturbed exponential model accounting for perturbations from repeated test flashes (Method 3)
A unified model that describes the data as a function of experimental acquisition time ($t_{acq}$), simultaneously capturing the intrinsic exponential recovery in the dark and the discrete perturbations induced by sequential probing flashes. The model iterates through two steps:

**Step 1: Multiplicative perturbation by a probing flash at time $t_{i-1}$:**
$$\Delta Y(t_{i-1})=Y(t_{i-1})\times\eta$$
$$Y^{\prime}(t_{i-1})=Y(t_{i-1})-\Delta Y(t_{i-1})$$

**Step 2: Exponential recovery between measurement times $t_{i-1}$ and $t_{i}$:**
$$Y(t_{i})=Y_{sat}-(Y_{sat}-Y^{\prime}(t_{i-1}))e^{-(t_{i}-t_{i-1})/\tau}$$

The parameters ($\tau$, $\eta$, $Y_{sat}$, $Y_{0}$) are optimized globally by minimizing the RMS error.


---

## Repository Structure & Figure Mapping

The scripts map directly to the figures presented in the manuscript. All scripts rely on the local configuration file `plot_configuration_manuscript.py` to ensure consistent aesthetics.

| Script Name | Description & Manuscript Figure |
| :--- | :--- |
| `plot_configuration_manuscript.py` | Global configuration file for standardizing matplotlib aesthetics. |
| `fig1_fig2_org_responses_visualizations.py` | Visualize ORG response traces. **(Generates Fig. 1 & Fig. 2)** |
| `fig3_fig4_org_vs_tadapt_exp_fits.py` | Fits ORG amplitude and dynamic parameters to the _exponential model_ (Method 1), as functions of dark adaptation time $t_{adapt}$. **(Generates Fig. 3 & Fig. 4)** |
| `fig5_fig6_org_vs_tacq_prob_fits.py` | Fits ORG amplitude and dynamic parameters to the probabilistic model (Method 2), as functions of acquisition time $t_{acq}$. **(Generates Fig. 5 & Fig. 6)** |
| `fig7_protocol_bleach_simu.py` | Simulates the experimental timeline and theoretical pigment regeneration with test bleaches. **(Generates Fig. 7)** |
| `fig8_org_vs_tadapt_RL_fits.py` | Fits ORG amplitude and dynamic parameters to the _Rate-limited (RL) model_, as functions of dark adaptation time $t_{adapt}$. **(Generates Fig. 8)** |

### Data Directory (`/data/`)
Processed ORG paramters collected from 8-10 repeated trails from three subjects, extracted using an overdamped RLC model fit, $$\Delta OPL_{fit} = A_1(-e^{-t/\tau_a} + e^{-t/\tau_b})$$, further details can be found in the manuscript.
The data files are required to run the fitting scripts. Ensure the following files are located in a `/data/` subdirectory relative to the scripts:
* `ORG_curves_304s_example.npz`
* `fitRMS30_sub[1/2]_rep[X]_plotdata.pkl`
* `sub[1/2/3]_median_data.csv`
* `sub[1/2/3]_combined_data.csv`

The `.csv` datasets (`_combined_data` and `_median_data`) contain the following variables:
#### Variables related to the experimental protocol
* **`datalabel`**: The dark adaptation trial number # (e.g., `DArep10`).
* **`tacq`**: Corresponds to $t_{acq}$. The acquisition time (in seconds) indicating exactly when the 2.1-second OCT volume recording and probing flash occurred relative to the initial bleaching flash.
* **`Tall`**: A cumulative list/array of the exact acquisition times (in seconds) that all previous probing flashes were delivered prior to the current measurement. Used to iteratively calculate fractional pigment loss.
* **`bleaching`**: The calculated percentage of photopigment bleached (or remaining) at that specific acquisition time. 
* **`cell index`** *(Only in `_combined_data.csv`)*: The index of each single segmented cone within the $1^{\circ}\times1^{\circ}$ field of view.

#### Primary ORG parameters
* **`opl0406avg`**: Corresponds to **$\Delta OPL_{max}$**. The maximum optical path length change of the cone outer segment, calculated directly from the raw measurements by averaging the five highest $\Delta OPL$ values.
* **`OPL_pk_fit`**: Corresponds to **$\Delta OPL_{fitting,max}$**. The peak amplitude of the outer segment elongation derived mathematically from the overdamped RLC model fit.
* **`t_pk`**: Corresponds to **$t_{pk}$**. The time-to-peak (in seconds), by solving t for the first zero of the RLC model’s derivative d($\Delta OPL_{fit}$)/dt = 0.
* **`tau_b_fit`**: Corresponds to **$$\tau_{b}$$**. The late contraction rate (in $s^{-1}$) of the outer segment, describing the late, slow contraction rate of the outer segment.

#### Intermediate ORG model parameters
*(Found primarily in `_combined_data.csv`)*
These variables are the underlying coefficients generated when fitting the individual cone responses to the overdamped RLC model ($$\Delta OPL_{fit} = A_1(-e^{-t/\tau_a} + e^{-t/\tau_b})$$).
* **`A_1_fit`**: The amplitude scaling factor ($A_1$) of the RLC model.
* **`tau_a_fit`**: The time constant ($\tau_a$) governing the rapid rising phase (elongation) of the cone outer segment response.


## Requirements & Installation

The analysis is written in Python 3. To run the scripts locally, install the required libraries:

```bash
pip install numpy scipy pandas matplotlib
