"""
Critique of Anthropic Economic Index: Simple OLS Analysis
==========================================================
Reproduces their OLS regression and runs diagnostics.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import levene
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['MPLBACKEND'] = 'Agg'

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(
    "/Users/ilanstrauss/anthropic-econ-critique/data/release_2025_09_15/data/output/aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv",
    keep_default_na=False,
    na_values=[""]
)

# Extract country-level data
country = df[(df['geography'] == 'country') & (df['facet'] == 'country')]

# Pivot to wide format
country_wide = country.pivot_table(
    index=['geo_id', 'geo_name'],
    columns='variable',
    values='value',
    aggfunc='first'
).reset_index()

# Filter to countries with sufficient data
MIN_OBS = 200

usage_counts = country_wide[['geo_id', 'usage_count']].dropna()
filtered_countries = usage_counts[usage_counts['usage_count'] >= MIN_OBS]['geo_id'].tolist()

# Create analysis dataset
analysis_df = country_wide[country_wide['geo_id'].isin(filtered_countries)].copy()
analysis_df = analysis_df[['geo_id', 'geo_name', 'gdp_per_working_age_capita',
                            'usage_per_capita_index', 'usage_count', 'working_age_pop']].dropna()

# Log transform
analysis_df['log_gdp'] = np.log(analysis_df['gdp_per_working_age_capita'])
analysis_df['log_usage'] = np.log(analysis_df['usage_per_capita_index'])

# ============================================================
# 2. REPRODUCE THEIR OLS REGRESSION
# ============================================================

X = sm.add_constant(analysis_df['log_gdp'])
y = analysis_df['log_usage']

ols_model = sm.OLS(y, X).fit()

# Calculate residuals
analysis_df['ols_resid'] = ols_model.resid
analysis_df['ols_predicted'] = ols_model.predict(X)

# ============================================================
# 3. IDENTIFY PROBLEMATIC FITS
# ============================================================

# Countries with largest residuals
analysis_df['abs_resid'] = np.abs(analysis_df['ols_resid'])
outliers = analysis_df.nlargest(10, 'abs_resid')[['geo_id', 'geo_name', 'log_gdp', 'log_usage', 'ols_resid', 'usage_count']]

# Leverage analysis
X_array = X.values
hat_matrix = X_array @ np.linalg.inv(X_array.T @ X_array) @ X_array.T
leverage = np.diag(hat_matrix)
analysis_df['leverage'] = leverage
analysis_df['cooks_d'] = (analysis_df['ols_resid']**2 / (2 * ols_model.mse_resid)) * (leverage / (1 - leverage)**2)

high_influence = analysis_df.nlargest(10, 'cooks_d')[['geo_id', 'geo_name', 'cooks_d', 'leverage', 'ols_resid']]

# ============================================================
# 4. HETEROGENEITY TESTS
# ============================================================

# Split by GDP terciles
terciles = pd.qcut(analysis_df['log_gdp'], 3, labels=['Low', 'Mid', 'High'])
groups = [analysis_df[terciles == t]['ols_resid'] for t in ['Low', 'Mid', 'High']]

# Levene's test for heteroscedasticity
levene_stat, levene_p = levene(*groups)

# Breusch-Pagan test
bp_stat, bp_pval, _, _ = het_breuschpagan(ols_model.resid, X)

# ============================================================
# 5. ROBUST REGRESSION COMPARISON
# ============================================================

rlm_model = RLM(y, X, M=sm.robust.norms.HuberT()).fit()

# ============================================================
# 6. SLOPES BY INCOME GROUP
# ============================================================

analysis_df['income_group'] = pd.qcut(analysis_df['log_gdp'], 3, labels=['Low', 'Mid', 'High'])

tercile_results = {}
for group in ['Low', 'Mid', 'High']:
    subset = analysis_df[analysis_df['income_group'] == group]
    if len(subset) > 3:
        X_sub = sm.add_constant(subset['log_gdp'])
        model_sub = sm.OLS(subset['log_usage'], X_sub).fit()
        tercile_results[group] = {
            'slope': model_sub.params['log_gdp'],
            'se': model_sub.bse['log_gdp'],
            'n': len(subset)
        }

# ============================================================
# 7. SUMMARY TABLE
# ============================================================

results_list = [
    {'Method': 'Anthropic OLS', 'Slope': ols_model.params['log_gdp'], 'SE': ols_model.bse['log_gdp']},
    {'Method': 'Robust (Huber)', 'Slope': rlm_model.params['log_gdp'], 'SE': rlm_model.bse['log_gdp']},
]

for group in ['Low', 'Mid', 'High']:
    if group in tercile_results:
        results_list.append({
            'Method': f'{group} GDP countries',
            'Slope': tercile_results[group]['slope'],
            'SE': tercile_results[group]['se']
        })

summary_table = pd.DataFrame(results_list)

# ============================================================
# 8. SAVE RESULTS
# ============================================================

analysis_df.to_csv('/Users/ilanstrauss/anthropic-econ-critique/analysis_results.csv', index=False)
