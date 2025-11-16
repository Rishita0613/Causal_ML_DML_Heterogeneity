# ----------------------------------------------------------------------
# GitHub Project 1: Causal Machine Learning
# Application: Double Machine Learning (DML) for Causal Heterogeneity
# Author: [Your Name]
# Date: November 2025
# ----------------------------------------------------------------------
# This script simulates a dataset where the effect of a treatment (e.g.,
# a discount campaign) varies based on customer characteristics (X).
# It uses the Causal Forest DML framework from the econml
# library to estimate the Average Treatment Effect (ATE) and the
# Conditional Average Treatment Effect (CATE).
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML

# --- 1. CONFIGURATION AND DATA SIMULATION ---
N_SAMPLES = 5000
N_FEATURES = 5
np.random.seed(42)

print("1. Simulating Synthetic Causal Data...")

# X: Confounding Features (Customer Characteristics)
X = np.random.normal(0, 1, size=(N_SAMPLES, N_FEATURES))

# W: Treatment Variable (e.g., Exposure to a Marketing Campaign, 0 or 1)
# Treatment assignment depends on features X (Requires Causal ML)
W = np.random.binomial(1, p=1 / (1 + np.exp(X[:, 0] - X[:, 1] + np.random.normal(0, 0.5, N_SAMPLES))))
# Reshape W to a column vector (N, 1) for initial data consistency
W = W.reshape(-1, 1)

# --- Define Causal Heterogeneity Function (True Effect) ---
def true_treatment_effect(X):
    # Effect is stronger for customers with high X[:, 2]
    return 1 + 0.5 * X[:, 2] + 0.3 * np.sin(X[:, 3])

# Y: Outcome Variable (e.g., Customer Spend / Conversion Rate)
Y_baseline = 5 + 2 * X[:, 0] - 1.5 * X[:, 4]
Y_treatment_effect = true_treatment_effect(X) * W.flatten()
Y = Y_baseline + Y_treatment_effect + np.random.normal(0, 0.5, N_SAMPLES)
# Reshape Y to a column vector (N, 1) for initial data consistency
Y = Y.reshape(-1, 1)

print(f"   -> Data Generated: {N_SAMPLES} samples.")

# --- 2. CAUSAL FOREST DML SETUP ---
print("\n2. Setting up Causal Forest DML Estimator...")

# Define models for the nuisance functions
model_Y = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
model_W = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)

# Initialize the DML estimator
dml_estimator = CausalForestDML(
    model_y=model_Y,
    model_t=model_W,
    criterion='mse',
    cv=3,
    random_state=42
)

# --- 3. TRAINING AND ESTIMATION (ATE) & CATE ---
print("3. Training DML to estimate Average Treatment Effect (ATE) and CATE...")

# FIX: Use .ravel() to convert (N, 1) vectors to 1D arrays (N,) 
# to satisfy scikit-learn/econml input requirements and suppress the warning.
dml_estimator.fit(Y=Y.ravel(), T=W.ravel(), X=X)

# Estimate the Average Treatment Effect (ATE).
ate_result = dml_estimator.ate_inference(X=X)

# FIX: ATE Estimate
ate_estimate = np.squeeze(ate_result.value)

# FIX: Confidence Interval - conf_int_mean() is called to return the 1D array of bounds.
ate_ci = ate_result.conf_int_mean() 

# NEW FIX: P-value - The pvalue() method must be called to return the scalar value.
ate_p_value = np.squeeze(ate_result.pvalue())


print("-" * 60)
print("DML Result: Average Treatment Effect (ATE)")
print("-" * 60)
print(f"Estimated ATE: {ate_estimate:.4f}")
print(f"95% CI:        ({ate_ci[0]:.4f}, {ate_ci[1]:.4f})")
print(f"P-value:       {ate_p_value:.4f}")
print("-" * 60)

# --- 4. HETEROGENEITY ANALYSIS (CATE) ---
print("\n4. Estimating Conditional Average Treatment Effect (CATE)...")

# Predict the treatment effect (tau) for a range of X values.
# Sort by the key feature X[:, 2] for smooth plotting
X_for_cate_plot = X[np.argsort(X[:, 2]), :]
cate_estimates = dml_estimator.effect(X_for_cate_plot)

# --- 5. VISUALIZING HETEROGENEITY (CATE vs. ATE) ---
print("\n5. Generating Visualization of Causal Heterogeneity (CATE)...")

plt.figure(figsize=(10, 6))
# Plot the CATE estimates against the feature that drives heterogeneity (X[:, 2])
plt.scatter(X_for_cate_plot[:, 2], cate_estimates, s=5, alpha=0.5, label='Estimated CATE (Personalized Effect)')

# Plot the Estimated ATE
plt.axhline(ate_estimate, color='red', linestyle='--', linewidth=2, label=f'Estimated ATE ({ate_estimate:.2f})')

# Plot the TRUE causal effect (for comparison with the simulated truth)
true_effects = true_treatment_effect(X_for_cate_plot)
plt.plot(X_for_cate_plot[:, 2], true_effects, color='orange', linewidth=2, label='True Causal Effect (Simulated)')


plt.title('Causal Heterogeneity: CATE vs. ATE for Personalized Policy')
plt.xlabel('Customer Characteristic (Feature X[:, 2] - e.g., Engagement Score)')
plt.ylabel('Treatment Effect on Outcome (tau)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()

print("\nAnalysis Conclusion:")
print("The ATE is statistically significant, but the CATE plot reveals a large spread in effect, proving heterogeneity.")
print("This supports a personalized policy where treatment is prioritized for individuals with high values of X[:, 2].")
print("-" * 60)