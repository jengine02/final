# Signal Detection Theory and Delta Plot Analysis

This project analyzes simulated response time data from a 2×2×2 factorial experiment using two modeling approaches: Signal Detection Theory (SDT) and delta plots (used to examine decision-making dynamics similar to diffusion decision models).

## Data Description

The dataset (`data.csv`) includes trial-by-trial data for multiple participants across different experimental conditions. Each trial varies along three factors:

- **Trial Difficulty**: Easy or Hard
- **Stimulus Type**: Simple or Complex
- **Signal Presence**: Present or Absent

Each row contains:
- Participant ID
- Trial number
- RT (response time)
- Choice and correctness
- Condition labels (difficulty, stimulus type, signal)

## Analysis Summary

### 1. Signal Detection Theory (SDT)

We use a hierarchical Bayesian SDT model to estimate:

- **d′ (discriminability)**: how well participants distinguish signal from noise.
- **c (criterion)**: the decision threshold or bias in responding.

The model is implemented in PyMC and includes both participant-level and group-level parameters. We check model convergence using:
- R-hat values (should be close to 1.00)
- Effective sample size
- Trace plots and posterior summaries

### 2. Delta Plot Analysis

Delta plots are generated to analyze how RTs differ across conditions. They show RT percentiles (10th to 90th) for:
- All trials
- Correct trials
- Error trials

This allows us to examine how trial difficulty and stimulus complexity affect decision dynamics, such as RT variability and shifts in decision speed.

## Files Included

- `sdt_ddm.py`: Contains all code for loading data, fitting the SDT model, and generating delta plots
- `data.csv`: The response time dataset
- `output/`: Directory where delta plots are saved
- `README.md`: This file

## How to Run the Analysis

You will need Python with the following packages installed: `pandas`, `numpy`, `pymc`, `arviz`, and `matplotlib`.

# Load and prepare SDT data
sdt_data = read_data("data.csv", prepare_for="sdt")
model = apply_hierarchical_sdt_model(sdt_data)

import pymc as pm
with model:
    trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9)

# Load data and generate delta plots for participant 1
dp_data = read_data("data.csv", prepare_for="delta plots")
draw_delta_plots(dp_data, pnum=1)

## Credits
ChatGPT was utilized in order to edit coding errors and help organize this README file.