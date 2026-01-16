"""
Critique of Anthropic Economic Index: Partial Pooling vs OLS
=============================================================
Uses statsmodels MixedLM for partial pooling (random effects).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.robust.robust_linear_model import RLM
import warnings
warnings.filterwarnings('ignore')

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

# Add income group for partial pooling
analysis_df['income_group'] = pd.qcut(analysis_df['log_gdp'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# ============================================================
# 2. REPRODUCE THEIR OLS REGRESSION
# ============================================================

X = sm.add_constant(analysis_df['log_gdp'])
y = analysis_df['log_usage']
ols_model = sm.OLS(y, X).fit()

analysis_df['ols_resid'] = ols_model.resid
analysis_df['ols_predicted'] = ols_model.predict(X)

# ============================================================
# 3. PARTIAL POOLING: MIXED EFFECTS MODEL
# ============================================================

# Random intercepts by income group
model_ri = MixedLM.from_formula(
    'log_usage ~ log_gdp',
    groups='income_group',
    data=analysis_df
)
result_ri = model_ri.fit()

# ============================================================
# 4. PARTIAL POOLING: RANDOM SLOPES MODEL
# ============================================================

# Random slopes by income group
model_rs = MixedLM.from_formula(
    'log_usage ~ log_gdp',
    groups='income_group',
    re_formula='~log_gdp',
    data=analysis_df
)
result_rs = model_rs.fit()

# ============================================================
# 5. COMPARISON OF EFFECT SIZES
# ============================================================

ols_slope = ols_model.params['log_gdp']
ols_se = ols_model.bse['log_gdp']
pp_slope = result_rs.fe_params['log_gdp']
pp_se = result_rs.bse_fe['log_gdp']

# ============================================================
# 6. OUTLIERS AND INFLUENTIAL OBSERVATIONS
# ============================================================

analysis_df['abs_resid'] = np.abs(analysis_df['ols_resid'])
X_array = X.values
hat_matrix = X_array @ np.linalg.inv(X_array.T @ X_array) @ X_array.T
analysis_df['leverage'] = np.diag(hat_matrix)
analysis_df['cooks_d'] = (analysis_df['ols_resid']**2 / (2 * ols_model.mse_resid)) * (analysis_df['leverage'] / (1 - analysis_df['leverage'])**2)

outliers = analysis_df.nlargest(10, 'abs_resid')[['geo_id', 'geo_name', 'gdp_per_working_age_capita', 'usage_per_capita_index', 'ols_resid']]

threshold = 4 / len(analysis_df)
high_influence = analysis_df[analysis_df['cooks_d'] > threshold][['geo_id', 'geo_name', 'cooks_d']].sort_values('cooks_d', ascending=False)

# ============================================================
# 7. ROBUST REGRESSION
# ============================================================

rlm_model = RLM(y, X, M=sm.robust.norms.HuberT()).fit()

# ============================================================
# 8. SUBSAMPLE ANALYSIS
# ============================================================

analysis_df['income_tercile'] = pd.qcut(analysis_df['log_gdp'], 3, labels=['Low', 'Mid', 'High'])

tercile_results = {}
for group in ['Low', 'Mid', 'High']:
    subset = analysis_df[analysis_df['income_tercile'] == group]
    if len(subset) > 5:
        X_sub = sm.add_constant(subset['log_gdp'])
        model_sub = sm.OLS(subset['log_usage'], X_sub).fit()
        tercile_results[group] = {
            'slope': model_sub.params['log_gdp'],
            'se': model_sub.bse['log_gdp'],
            'n': len(subset)
        }

# ============================================================
# 9. SUMMARY TABLE
# ============================================================

summary_data = [
    {'Method': 'Anthropic OLS (their claim)', 'Slope': 0.70, 'Notes': 'Published estimate'},
    {'Method': 'Reproduced OLS', 'Slope': ols_slope, 'Notes': f'SE={ols_se:.3f}'},
    {'Method': 'Partial Pooling (avg)', 'Slope': pp_slope, 'Notes': f'SE={pp_se:.3f}'},
    {'Method': 'Robust Regression', 'Slope': rlm_model.params['log_gdp'], 'Notes': 'Downweights outliers'},
]

# Add per-quintile
for group, effect in result_rs.random_effects.items():
    slope_dev = effect.iloc[1] if len(effect) > 1 else 0
    group_slope = result_rs.fe_params['log_gdp'] + slope_dev
    summary_data.append({'Method': f'  Quintile {group}', 'Slope': group_slope, 'Notes': 'Partial pooling'})

summary_df = pd.DataFrame(summary_data)

# ============================================================
# 10. SAVE RESULTS
# ============================================================

analysis_df.to_csv('/Users/ilanstrauss/anthropic-econ-critique/analysis_results.csv', index=False)
