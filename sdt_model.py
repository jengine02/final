import pandas as pd
data = pd.read_csv("data.csv")

from sdt_ddm import read_data, apply_hierarchical_sdt_model
sdt_data = read_data("data.csv", prepare_for='sdt', display=True)
model = apply_hierarchical_sdt_model(sdt_data)

import pymc as pm
with model:
    trace = pm.sample(1000, tune=1000, target_accept=0.9, chains=2, return_inferencedata=True)

import arviz as az
az.summary(trace)

from sdt_ddm import draw_delta_plots
dp_data = read_data("data.csv", prepare_for="delta plots")
draw_delta_plots(dp_data, pnum=1)  # You can loop through other pnums too