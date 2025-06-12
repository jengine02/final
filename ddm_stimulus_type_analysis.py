import pandas as pd
import numpy as np
import hddm

# Load your data
data = pd.read_csv("data.csv")

# Filter for Participant 1
data_p1 = data[data['participant_id'] == 1].copy()

# HDDM expects 'response' to be 1 (upper boundary) or 0 (lower boundary)
# If your 'accuracy' column is 1 for correct and 0 for error, you can use it directly
data_p1['response'] = data_p1['accuracy']

# Fit HDDM model with all parameters varying by stimulus_type
model = hddm.HDDM(data_p1, depends_on={'v': 'stimulus_type', 'a': 'stimulus_type', 't': 'stimulus_type', 'z': 'stimulus_type'})
model.find_starting_values()
model.sample(1000, burn=200)

# Print posterior means for each parameter by stimulus_type
print(model.nodes_db.loc[model.nodes_db['stochastic'], ['node', 'mean', 'std']])

v_simple = model.nodes_db.node['v(simple)'].trace()
v_complex = model.nodes_db.node['v(complex)'].trace()
diff = v_complex - v_simple
print(f"Mean difference: {diff.mean():.2f}")
print(f"95% CI: [{np.percentile(diff, 2.5):.2f}, {np.percentile(diff, 97.5):.2f}]")