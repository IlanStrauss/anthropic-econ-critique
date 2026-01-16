"""
Critique of Anthropic Economic Index: Partial Pooling vs OLS
=============================================================
Uses statsmodels MixedLM for partial pooling (random effects).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

# Load enriched data
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

for group, effect in result_ri.random_effects.items():

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

for group, effect in result_rs.random_effects.items():
    # effect is a Series with index like ['Group', 'log_gdp'] or ['Intercept', 'log_gdp']
    slope_dev = effect.iloc[1] if len(effect) > 1 else 0
    group_slope = result_rs.fe_params['log_gdp'] + slope_dev

# ============================================================
# 5. COMPARISON OF EFFECT SIZES
# ============================================================

ols_slope = ols_model.params['log_gdp']
ols_se = ols_model.bse['log_gdp']
pp_slope = result_rs.fe_params['log_gdp']
pp_se = result_rs.bse_fe['log_gdp']

# Per-quintile slopes
for group, effect in result_rs.random_effects.items():
    slope_dev = effect.iloc[1] if len(effect) > 1 else 0
    group_slope = result_rs.fe_params['log_gdp'] + slope_dev

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

from statsmodels.robust.robust_linear_model import RLM
rlm_model = RLM(y, X, M=sm.robust.norms.HuberT()).fit()

# ============================================================
# 8. SUBSAMPLE ANALYSIS
# ============================================================

analysis_df['income_tercile'] = pd.qcut(analysis_df['log_gdp'], 3, labels=['Low', 'Mid', 'High'])

for group in ['Low', 'Mid', 'High']:
    subset = analysis_df[analysis_df['income_tercile'] == group]
    if len(subset) > 5:
        X_sub = sm.add_constant(subset['log_gdp'])
        model_sub = sm.OLS(subset['log_usage'], X_sub).fit()

# ============================================================
# 9. POLICY IMPLICATIONS
# ============================================================

ANTHROPIC'S CLAIM: "1% increase in GDP → 0.7% increase in AI usage"

CRITIQUE FINDINGS:
==================

1. SLOPE VARIES BY INCOME LEVEL
   - Low income countries:  ~0.76 (HIGHER than global average)
   - Mid income countries:  ~0.44 (MUCH LOWER than global average)
   - High income countries: ~0.63 (LOWER than global average)

   The global 0.7 average MASKS huge heterogeneity.

2. ROBUST REGRESSION GIVES LOWER SLOPE (~0.67)
   - Outliers like Israel inflate the OLS estimate
   - After downweighting outliers, effect is slightly smaller

3. INFLUENTIAL OBSERVATIONS
   - Small rich countries (Qatar, Kuwait, Israel) have outsized influence
   - African countries (Angola, Tanzania) are strong outliers

POLICY IMPLICATIONS:
====================

A. FOR LOW-INCOME COUNTRIES:
   - GDP growth IS strongly associated with AI adoption
   - Economic development may be necessary condition for AI uptake
   - But causality unclear: does GDP enable AI or vice versa?

B. FOR MIDDLE-INCOME COUNTRIES:
   - Weakest relationship between GDP and AI usage
   - Suggests OTHER factors dominate: education, English proficiency,
     tech infrastructure, regulatory environment
   - Policy should focus on these, not just GDP growth

C. FOR HIGH-INCOME COUNTRIES:
   - AI adoption varies widely even at similar GDP levels
   - Israel is massive over-adopter, Gulf states are under-adopters
   - Cultural/linguistic/policy factors likely dominate

D. GENERAL:
   - Anthropic's simple "GDP predicts usage" story is oversimplified
   - Country-specific interventions needed
   - One-size-fits-all policy recommendations are inappropriate
""")

# ============================================================
# 10. SUMMARY TABLE
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
# 11. SAVE RESULTS
# ============================================================

analysis_df.to_csv('/Users/ilanstrauss/anthropic-econ-critique/analysis_results.csv', index=False)

