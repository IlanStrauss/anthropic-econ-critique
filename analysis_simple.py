"""
Critique of Anthropic Economic Index: Partial Pooling vs OLS
=============================================================
Reproduces their OLS regression and compares to hierarchical model.
Goal: See if key effect sizes change with proper uncertainty quantification.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Don't import matplotlib yet - it slows things down
import os
os.environ['MPLBACKEND'] = 'Agg'

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

# Filter to countries with sufficient data
MIN_OBS = 200  # Their threshold

# Get usage counts
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

# Their claim: "a 1% increase in GDP per capita correlates with 0.7% increased usage per capita"

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
# 4. PARTIAL POOLING MODEL (HIERARCHICAL)
# ============================================================

try:
    import bambi as bmb
    import arviz as az

    # Random intercepts model: each country can have different baseline usage
    # but shares common GDP slope

    # Prepare data for bambi
    model_data = analysis_df[['geo_id', 'log_gdp', 'log_usage']].copy()

    # Model: log_usage ~ log_gdp + (1 | geo_id)
    # This is partial pooling on intercepts
    model = bmb.Model("log_usage ~ log_gdp + (1 | geo_id)", model_data)

    # Fit with variational inference (faster than MCMC)
    results = model.fit(method="advi", n=30000, random_seed=42)

    # Extract posterior summaries
    summary = az.summary(results, var_names=['log_gdp', 'Intercept'])

    # Compare slopes
    partial_slope_mean = summary.loc['log_gdp', 'mean']
    partial_slope_sd = summary.loc['log_gdp', 'sd']

    # Extract random effects (country-specific intercepts)
    re_summary = az.summary(results, var_names=['1|geo_id'])
    re_summary['geo_id'] = [x.split('[')[1].rstrip(']') for x in re_summary.index]
    top_positive = re_summary.nlargest(10, 'mean')[['geo_id', 'mean', 'sd']]

    top_negative = re_summary.nsmallest(10, 'mean')[['geo_id', 'mean', 'sd']]

    PARTIAL_POOLING_SUCCESS = True

except Exception as e:
    import traceback
    traceback.print_exc()
    PARTIAL_POOLING_SUCCESS = False

# ============================================================
# 5. HETEROGENEITY TESTS
# ============================================================

# Test for heteroscedasticity
from scipy.stats import levene

# Split by GDP terciles
terciles = pd.qcut(analysis_df['log_gdp'], 3, labels=['Low', 'Mid', 'High'])
groups = [analysis_df[terciles == t]['ols_resid'] for t in ['Low', 'Mid', 'High']]

# Levene's test
stat, p = levene(*groups)
if p < 0.05:

# Breusch-Pagan test
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(ols_model.resid, X)

# ============================================================
# 6. ROBUST REGRESSION COMPARISON
# ============================================================

# Robust regression (Huber)
from statsmodels.robust.robust_linear_model import RLM

rlm_model = RLM(y, X, M=sm.robust.norms.HuberT()).fit()

# Compare

# ============================================================
# 7. SLOPES BY INCOME GROUP
# ============================================================

analysis_df['income_group'] = pd.qcut(analysis_df['log_gdp'], 3, labels=['Low', 'Mid', 'High'])

for group in ['Low', 'Mid', 'High']:
    subset = analysis_df[analysis_df['income_group'] == group]
    if len(subset) > 3:
        X_sub = sm.add_constant(subset['log_gdp'])
        model_sub = sm.OLS(subset['log_usage'], X_sub).fit()

# ============================================================
# 8. SUMMARY TABLE
# ============================================================

results_list = [
    {'Method': 'Anthropic OLS', 'Slope': ols_model.params['log_gdp'], 'SE': ols_model.bse['log_gdp']},
    {'Method': 'Robust (Huber)', 'Slope': rlm_model.params['log_gdp'], 'SE': rlm_model.bse['log_gdp']},
]

for group in ['Low', 'Mid', 'High']:
    subset = analysis_df[analysis_df['income_group'] == group]
    if len(subset) > 3:
        X_sub = sm.add_constant(subset['log_gdp'])
        model_sub = sm.OLS(subset['log_usage'], X_sub).fit()
        results_list.append({
            'Method': f'{group} GDP countries',
            'Slope': model_sub.params['log_gdp'],
            'SE': model_sub.bse['log_gdp']
        })

summary_table = pd.DataFrame(results_list)

# ============================================================
# 9. POLICY IMPLICATIONS
# ============================================================

KEY FINDINGS:
-------------
1. Anthropic uses complete pooling OLS, ignoring country heterogeneity

2. High-influence points (outliers like Israel, small rich countries)
   disproportionately affect the global slope estimate

3. The 0.7 elasticity claim ("1% increase in GDP → 0.7% increase in usage")
   is a global average that masks significant country-level variation

POLICY IMPLICATIONS:
--------------------
If partial pooling shows smaller effect sizes:
- Anthropic may be OVERSTATING how much GDP drives AI adoption
- Other factors (education, tech infrastructure, language) may matter more
- Policy focus on GDP growth alone may be misguided

If slopes vary significantly by country/region:
- One-size-fits-all policy recommendations are inappropriate
- Need country-specific interventions
- Some countries are "under-adopters" relative to GDP (policy opportunity)
- Some countries are "over-adopters" (different adoption drivers)
""")

# Save results
analysis_df.to_csv('/Users/ilanstrauss/anthropic-econ-critique/analysis_results.csv', index=False)
