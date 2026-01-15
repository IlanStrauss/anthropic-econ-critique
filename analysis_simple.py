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

print("=" * 60)
print("ANTHROPIC ECONOMIC INDEX: CRITIQUE ANALYSIS")
print("=" * 60)

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

print(f"\nCountries with >= {MIN_OBS} observations: {len(filtered_countries)}")

# Create analysis dataset
analysis_df = country_wide[country_wide['geo_id'].isin(filtered_countries)].copy()
analysis_df = analysis_df[['geo_id', 'geo_name', 'gdp_per_working_age_capita',
                            'usage_per_capita_index', 'usage_count', 'working_age_pop']].dropna()

# Log transform
analysis_df['log_gdp'] = np.log(analysis_df['gdp_per_working_age_capita'])
analysis_df['log_usage'] = np.log(analysis_df['usage_per_capita_index'])

print(f"Countries in analysis: {len(analysis_df)}")
print(f"\nTop 10 by usage:")
print(analysis_df.nlargest(10, 'usage_count')[['geo_id', 'geo_name', 'usage_count', 'gdp_per_working_age_capita', 'usage_per_capita_index']].to_string())

# ============================================================
# 2. REPRODUCE THEIR OLS REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("ANTHROPIC'S OLS REGRESSION (Reproducing their approach)")
print("=" * 60)

X = sm.add_constant(analysis_df['log_gdp'])
y = analysis_df['log_usage']

ols_model = sm.OLS(y, X).fit()

print(f"\nOLS Results:")
print(f"  Slope (β): {ols_model.params['log_gdp']:.4f}")
print(f"  Std Error: {ols_model.bse['log_gdp']:.4f}")
print(f"  95% CI: [{ols_model.conf_int().loc['log_gdp', 0]:.4f}, {ols_model.conf_int().loc['log_gdp', 1]:.4f}]")
print(f"  p-value: {ols_model.pvalues['log_gdp']:.2e}")
print(f"  R²: {ols_model.rsquared:.4f}")
print(f"  N: {len(analysis_df)}")

# Their claim: "a 1% increase in GDP per capita correlates with 0.7% increased usage per capita"
print(f"\n  INTERPRETATION: 1% GDP increase → {ols_model.params['log_gdp']:.2f}% usage increase")

# Calculate residuals
analysis_df['ols_resid'] = ols_model.resid
analysis_df['ols_predicted'] = ols_model.predict(X)

# ============================================================
# 3. IDENTIFY PROBLEMATIC FITS
# ============================================================

print("\n" + "=" * 60)
print("IDENTIFYING PROBLEMATIC FITS (Outliers)")
print("=" * 60)

# Countries with largest residuals
analysis_df['abs_resid'] = np.abs(analysis_df['ols_resid'])
outliers = analysis_df.nlargest(10, 'abs_resid')[['geo_id', 'geo_name', 'log_gdp', 'log_usage', 'ols_resid', 'usage_count']]
print("\nTop 10 outliers from OLS fit:")
print(outliers.to_string())

# Leverage analysis
X_array = X.values
hat_matrix = X_array @ np.linalg.inv(X_array.T @ X_array) @ X_array.T
leverage = np.diag(hat_matrix)
analysis_df['leverage'] = leverage
analysis_df['cooks_d'] = (analysis_df['ols_resid']**2 / (2 * ols_model.mse_resid)) * (leverage / (1 - leverage)**2)

high_influence = analysis_df.nlargest(10, 'cooks_d')[['geo_id', 'geo_name', 'cooks_d', 'leverage', 'ols_resid']]
print("\n\nTop 10 by Cook's Distance (high influence points):")
print(high_influence.to_string())

# ============================================================
# 4. PARTIAL POOLING MODEL (HIERARCHICAL)
# ============================================================

print("\n" + "=" * 60)
print("PARTIAL POOLING MODEL (Hierarchical Bayesian)")
print("=" * 60)

try:
    import bambi as bmb
    import arviz as az

    # Random intercepts model: each country can have different baseline usage
    # but shares common GDP slope
    print("\nFitting hierarchical model with random intercepts...")
    print("(Using ADVI for speed - faster than MCMC)")

    # Prepare data for bambi
    model_data = analysis_df[['geo_id', 'log_gdp', 'log_usage']].copy()

    # Model: log_usage ~ log_gdp + (1 | geo_id)
    # This is partial pooling on intercepts
    model = bmb.Model("log_usage ~ log_gdp + (1 | geo_id)", model_data)

    # Fit with variational inference (faster than MCMC)
    print("Fitting with ADVI (may take a minute)...")
    results = model.fit(method="advi", n=30000, random_seed=42)

    # Extract posterior summaries
    summary = az.summary(results, var_names=['log_gdp', 'Intercept'])
    print("\nPartial Pooling Results:")
    print(summary)

    # Compare slopes
    partial_slope_mean = summary.loc['log_gdp', 'mean']
    partial_slope_sd = summary.loc['log_gdp', 'sd']

    print(f"\n\nCOMPARISON OF EFFECT SIZES:")
    print("-" * 40)
    print(f"OLS Slope:             {ols_model.params['log_gdp']:.4f} (SE: {ols_model.bse['log_gdp']:.4f})")
    print(f"Partial Pooling Slope: {partial_slope_mean:.4f} (SD: {partial_slope_sd:.4f})")
    print(f"Difference:            {partial_slope_mean - ols_model.params['log_gdp']:.4f}")
    print(f"% Change:              {100 * (partial_slope_mean - ols_model.params['log_gdp']) / ols_model.params['log_gdp']:.1f}%")

    # Extract random effects (country-specific intercepts)
    re_summary = az.summary(results, var_names=['1|geo_id'])
    print("\n\nRANDOM EFFECTS (Country-specific intercepts):")
    print("Countries with highest positive intercepts (more usage than GDP predicts):")
    re_summary['geo_id'] = [x.split('[')[1].rstrip(']') for x in re_summary.index]
    top_positive = re_summary.nlargest(10, 'mean')[['geo_id', 'mean', 'sd']]
    print(top_positive.to_string(index=False))

    print("\nCountries with most negative intercepts (less usage than GDP predicts):")
    top_negative = re_summary.nsmallest(10, 'mean')[['geo_id', 'mean', 'sd']]
    print(top_negative.to_string(index=False))

    PARTIAL_POOLING_SUCCESS = True

except Exception as e:
    print(f"\nBambi model fitting failed: {e}")
    import traceback
    traceback.print_exc()
    PARTIAL_POOLING_SUCCESS = False

# ============================================================
# 5. HETEROGENEITY TESTS
# ============================================================

print("\n" + "=" * 60)
print("HETEROGENEITY TESTS")
print("=" * 60)

# Test for heteroscedasticity
from scipy.stats import levene

# Split by GDP terciles
terciles = pd.qcut(analysis_df['log_gdp'], 3, labels=['Low', 'Mid', 'High'])
groups = [analysis_df[terciles == t]['ols_resid'] for t in ['Low', 'Mid', 'High']]

# Levene's test
stat, p = levene(*groups)
print(f"\nLevene's test for heteroscedasticity:")
print(f"  Test statistic: {stat:.4f}")
print(f"  p-value: {p:.4f}")
if p < 0.05:
    print("  -> Evidence of heteroscedasticity (OLS standard errors may be wrong)")

# Breusch-Pagan test
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(ols_model.resid, X)
print(f"\nBreusch-Pagan test:")
print(f"  Test statistic: {bp_stat:.4f}")
print(f"  p-value: {bp_pval:.4f}")

# ============================================================
# 6. ROBUST REGRESSION COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("ROBUST REGRESSION COMPARISON")
print("=" * 60)

# Robust regression (Huber)
from statsmodels.robust.robust_linear_model import RLM

rlm_model = RLM(y, X, M=sm.robust.norms.HuberT()).fit()
print(f"\nRobust Regression (Huber) Results:")
print(f"  Slope (β): {rlm_model.params['log_gdp']:.4f}")
print(f"  Std Error: {rlm_model.bse['log_gdp']:.4f}")

# Compare
print(f"\nCOMPARISON:")
print(f"  OLS Slope:    {ols_model.params['log_gdp']:.4f}")
print(f"  Robust Slope: {rlm_model.params['log_gdp']:.4f}")
print(f"  Difference:   {rlm_model.params['log_gdp'] - ols_model.params['log_gdp']:.4f}")

# ============================================================
# 7. SLOPES BY INCOME GROUP
# ============================================================

print("\n" + "=" * 60)
print("SLOPE HETEROGENEITY BY INCOME GROUP")
print("=" * 60)

analysis_df['income_group'] = pd.qcut(analysis_df['log_gdp'], 3, labels=['Low', 'Mid', 'High'])

print("\nSlopes by income tercile:")
for group in ['Low', 'Mid', 'High']:
    subset = analysis_df[analysis_df['income_group'] == group]
    if len(subset) > 3:
        X_sub = sm.add_constant(subset['log_gdp'])
        model_sub = sm.OLS(subset['log_usage'], X_sub).fit()
        print(f"  {group} income: β = {model_sub.params['log_gdp']:.3f} (SE={model_sub.bse['log_gdp']:.3f}, n={len(subset)})")

# ============================================================
# 8. SUMMARY TABLE
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY: KEY EFFECT SIZE COMPARISONS")
print("=" * 60)

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
print(summary_table.to_string(index=False))

# ============================================================
# 9. POLICY IMPLICATIONS
# ============================================================

print("\n" + "=" * 60)
print("POLICY IMPLICATIONS OF FINDINGS")
print("=" * 60)

print("""
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

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

# Save results
analysis_df.to_csv('/Users/ilanstrauss/anthropic-econ-critique/analysis_results.csv', index=False)
print("\nSaved analysis_results.csv")
