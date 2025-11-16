# Causal_ML_DML_Heterogeneity
Project 1: Causal Machine Learning Toolkit - Heterogeneous Treatment Effect Estimation

Overview: Double Machine Learning (DML) for Optimal Firm Strategy

This project demonstrates the application of Double/Debiased Machine Learning (DML), specifically using a Causal Forest DML framework from the econml library, to estimate Heterogeneous Treatment Effects. This methodology is critical for advanced microeconomic analysis and firm policy evaluation, as it moves beyond the overall Average Treatment Effect (ATE) to uncover precise, segment-specific causal impacts (Conditional Average Treatment Effect, CATE).

Goal: Solving the Pricing Policy Problem

The project simulates a scenario where a firm implements a price policy change (Treatment) and must determine its effect on customer churn (Outcome). The challenge is that customer elasticity varies significantly based on usage and demographic features (X).

Traditional OLS regression is insufficient because it is vulnerable to regularisation bias and cannot accurately model the complex, non-linear relationships that drive heterogeneity. DML solves this by using Machine Learning (Random Forests) to control for confounding while ensuring the final causal estimate is unbiased.

Methodology: Causal Forest DML

Data Simulation: A synthetic dataset of 5,000 customers is generated where the true treatment effect is programmed to depend linearly on Feature X_2 (e.g., Customer Engagement Score).

DML Framework: The Causal Forest DML estimator is trained using Random Forest Regressors as the nuisance models (\hat{f}(X) and \hat{g}(X)).

Orthogonalisation: The process uses cross-fitting and residualization to isolate the true causal variation from the confounding variation.

CATE Estimation: The model predicts the unique causal effect ($\tau$) for every customer based on their characteristics.

Key Results and Policy Insight


Estimated ATE: 0.0000

95% CI:        (0.6747, 1.2446)

P-value:       0.0000


The analysis reveals a classic case of Masked Heterogeneity:

DML Result: Average Treatment Effect (ATE)

The estimated ATE across the entire customer base is near zero, suggesting a standard policy (treating everyone) would be ineffective.

Causal Heterogeneity (CATE) Visualisation

The CATE plot confirms that the overall ATE masks a wide range of individual effects.

Policy Recommendation: Treatment (e.g., the price hike) is highly effective/positive for customers where $X_2 > 0$ (high engagement) but neutral or negative for customers where X_2 < -2 (low engagement).

Conclusion: This validates the need for a personalised firm strategy, demonstrating that the DML methodology successfully identified the optimal segmentation for maximising policy returns.

Installation and Execution

The script requires Python 3.8+ and the following packages:

pip install numpy pandas scikit-learn matplotlib econml


To run the analysis and generate the plot:

python dml_heterogeneity_analysis.py
