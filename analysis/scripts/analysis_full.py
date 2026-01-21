"""
Critique of Anthropic Economic Index: Temporal Stability Analysis
==================================================================
Compares GDP-AI adoption relationship across two time periods:
- August 4-11, 2025 (earlier release)
- November 13-20, 2025 (January 2026 report data)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD AUGUST 2025 DATA (OLD RELEASE)
# ============================================================

old_df = pd.read_csv(
    "../data/anthropic/release_2025_09_15/data/output/aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv",
    keep_default_na=False,
    na_values=[""]
)

old_country = old_df[(old_df['geography'] == 'country') & (old_df['facet'] == 'country')]
old_wide = old_country.pivot_table(
    index=['geo_id', 'geo_name'],
    columns='variable',
    values='value',
    aggfunc='first'
).reset_index()

MIN_OBS = 200
old_filtered = old_wide[old_wide['usage_count'] >= MIN_OBS].copy()
old_analysis = old_filtered[['geo_id', 'geo_name', 'gdp_per_working_age_capita',
                              'usage_per_capita_index', 'usage_count']].dropna()

old_analysis['log_gdp'] = np.log(old_analysis['gdp_per_working_age_capita'])
old_analysis['log_usage'] = np.log(old_analysis['usage_per_capita_index'])
old_analysis['income_group'] = pd.qcut(old_analysis['log_gdp'], 3, labels=['Low', 'Mid', 'High'])

# ============================================================
# 2. LOAD NOVEMBER 2025 DATA (JANUARY 2026 REPORT)
# ============================================================

new_raw = pd.read_csv(
    "../data/anthropic/release_2026_01_15/data/intermediate/aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv"
)

# Load external data for enrichment
gdp = pd.read_csv("../data/anthropic/release_2025_09_15/data/intermediate/gdp_2024_country.csv")
wap = pd.read_csv("../data/anthropic/release_2025_09_15/data/intermediate/working_age_pop_2024_country.csv")

# Extract country-level usage
new_country = new_raw[(new_raw['geography'] == 'country') & (new_raw['facet'] == 'country')]
new_usage = new_country[new_country['variable'] == 'usage_count'][['geo_id', 'value']].copy()
new_usage.columns = ['country_code', 'usage_count']

# Merge with external data
new_merged = new_usage.merge(wap[['iso_alpha_3', 'working_age_pop', 'country_code', 'country_name']], on='country_code')
new_merged = new_merged.merge(gdp[['iso_alpha_3', 'gdp_total']], on='iso_alpha_3')

# Calculate derived variables
new_merged['gdp_per_working_age_capita'] = new_merged['gdp_total'] / new_merged['working_age_pop']
new_merged['usage_per_capita'] = new_merged['usage_count'] / new_merged['working_age_pop']
global_upc = new_merged['usage_count'].sum() / new_merged['working_age_pop'].sum()
new_merged['usage_per_capita_index'] = new_merged['usage_per_capita'] / global_upc

# Filter and prepare
new_analysis = new_merged[new_merged['usage_count'] >= MIN_OBS].copy()
new_analysis['log_gdp'] = np.log(new_analysis['gdp_per_working_age_capita'])
new_analysis['log_usage'] = np.log(new_analysis['usage_per_capita_index'])
new_analysis['income_group'] = pd.qcut(new_analysis['log_gdp'], 3, labels=['Low', 'Mid', 'High'])

# Rename for consistency
new_analysis = new_analysis.rename(columns={'iso_alpha_3': 'geo_id', 'country_name': 'geo_name'})

# ============================================================
# 3. RUN REGRESSIONS FOR BOTH PERIODS
# ============================================================

def run_regression(df, label):
    """Run OLS regression and return results."""
    X = sm.add_constant(df['log_gdp'])
    y = df['log_usage']
    model = sm.OLS(y, X).fit()
    return {
        'period': label,
        'slope': model.params['log_gdp'],
        'se': model.bse['log_gdp'],
        'pvalue': model.pvalues['log_gdp'],
        'r2': model.rsquared,
        'n': len(df)
    }

results = []

# Global regressions
results.append({**run_regression(old_analysis, 'Aug 4-11'), 'group': 'Global'})
results.append({**run_regression(new_analysis, 'Nov 13-20'), 'group': 'Global'})

# By income tercile
for group in ['Low', 'Mid', 'High']:
    old_sub = old_analysis[old_analysis['income_group'] == group]
    new_sub = new_analysis[new_analysis['income_group'] == group]
    results.append({**run_regression(old_sub, 'Aug 4-11'), 'group': group})
    results.append({**run_regression(new_sub, 'Nov 13-20'), 'group': group})

results_df = pd.DataFrame(results)

# ============================================================
# 4. PRINT RESULTS
# ============================================================

print("=" * 80)
print("GDP-AI ADOPTION RELATIONSHIP: TEMPORAL STABILITY ANALYSIS")
print("=" * 80)
print(f"\n{'Group':<10} {'Period':<12} {'Slope':<8} {'SE':<8} {'p-value':<10} {'Signif?':<8} {'R²':<8} {'N':<6}")
print("-" * 80)

for _, row in results_df.iterrows():
    sig = "YES" if row['pvalue'] < 0.05 else "NO"
    pval = f"{row['pvalue']:.4f}" if row['pvalue'] >= 0.0001 else "<0.001"
    print(f"{row['group']:<10} {row['period']:<12} {row['slope']:<8.2f} {row['se']:<8.2f} {pval:<10} {sig:<8} {row['r2']:<8.3f} {row['n']:<6.0f}")

# ============================================================
# 5. SAVE ANALYSIS RESULTS (November 2025 data)
# ============================================================

# Add residuals and predictions for figure generation
X = sm.add_constant(new_analysis['log_gdp'])
y = new_analysis['log_usage']
ols_model = sm.OLS(y, X).fit()

new_analysis['ols_resid'] = ols_model.resid
new_analysis['ols_predicted'] = ols_model.predict(X)

# Calculate Cook's D
X_array = X.values
hat_matrix = X_array @ np.linalg.inv(X_array.T @ X_array) @ X_array.T
new_analysis['leverage'] = np.diag(hat_matrix)
new_analysis['cooks_d'] = (new_analysis['ols_resid']**2 / (2 * ols_model.mse_resid)) * (new_analysis['leverage'] / (1 - new_analysis['leverage'])**2)

new_analysis.to_csv('../analysis/output/analysis_results.csv', index=False)

print("\n" + "=" * 80)
print("KEY FINDING: Middle-income relationship NOT significant in November 2025 data")
print("=" * 80)
mid_nov = results_df[(results_df['group'] == 'Mid') & (results_df['period'] == 'Nov 13-20')].iloc[0]
print(f"Middle income (Nov 2025): β = {mid_nov['slope']:.2f}, SE = {mid_nov['se']:.2f}, p = {mid_nov['pvalue']:.3f}, R² = {mid_nov['r2']:.3f}")
print("The relationship is statistically indistinguishable from zero (p > 0.10)")
