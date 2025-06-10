import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
from sdt_ddm import read_data

# Load data
sdt_data = read_data("data.csv", prepare_for="sdt")

# Convert to index arrays
pnum_idx = (sdt_data["pnum"].values - 1).astype(int)
cond_idx = sdt_data["condition"].values.astype(int)
n_signal = sdt_data["nSignal"].values
n_noise = sdt_data["nNoise"].values
hits = sdt_data["hits"].values
false_alarms = sdt_data["false_alarms"].values

with pm.Model() as model:
    C = sdt_data["condition"].nunique()
    
    # Group-level priors
    mean_d = pm.Normal("mean_d", mu=0.0, sigma=1.0, shape=C)
    sd_d = pm.HalfNormal("sd_d", sigma=1.0)
    mean_c = pm.Normal("mean_c", mu=0.0, sigma=1.0, shape=C)
    sd_c = pm.HalfNormal("sd_c", sigma=1.0)

    # Subject-level estimates (aligned with conditions)
    d = pm.Normal("d", mu=mean_d[cond_idx], sigma=sd_d, shape=len(sdt_data))
    c = pm.Normal("c", mu=mean_c[cond_idx], sigma=sd_c, shape=len(sdt_data))

    # Probabilities
    hit_rate = pm.Deterministic("hit_rate", pm.math.invlogit(d - c))
    fa_rate = pm.Deterministic("fa_rate", pm.math.invlogit(-c))

    # Likelihoods
    pm.Binomial("hits_obs", n=n_signal, p=hit_rate, observed=hits)
    pm.Binomial("fa_obs", n=n_noise, p=fa_rate, observed=false_alarms)

    # Sample
    trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9)

# Summary of posterior
summary = az.summary(trace, round_to=2)
print(summary)
