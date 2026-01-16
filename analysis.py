"""
Critique of Anthropic Economic Index: Full Analysis with Visualization
=======================================================================
Reproduces OLS regression, runs diagnostics, and creates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import levene
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
# 7. VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. OLS fit with residuals
ax1 = axes[0, 0]
colors = analysis_df['ols_resid'].apply(lambda x: 'red' if x > 0 else 'blue')
ax1.scatter(analysis_df['log_gdp'], analysis_df['log_usage'], c=colors, alpha=0.6, s=50)
x_line = np.linspace(analysis_df['log_gdp'].min(), analysis_df['log_gdp'].max(), 100)
ax1.plot(x_line, ols_model.params['const'] + ols_model.params['log_gdp'] * x_line,
         'k--', linewidth=2, label=f'OLS: β={ols_model.params["log_gdp"]:.3f}')
ax1.set_xlabel('ln(GDP per capita)')
ax1.set_ylabel('ln(AI Usage Index)')
ax1.set_title('Anthropic OLS Fit (Red=over, Blue=under predicted)')
ax1.legend()

# Label outliers
for _, row in analysis_df.nlargest(5, 'abs_resid').iterrows():
    ax1.annotate(row['geo_id'], (row['log_gdp'], row['log_usage']), fontsize=8)

# 2. Residual distribution
ax2 = axes[0, 1]
ax2.hist(analysis_df['ols_resid'], bins=20, edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--')
ax2.set_xlabel('OLS Residual')
ax2.set_ylabel('Count')
ax2.set_title('Residual Distribution (should be normal if OLS valid)')

# 3. Residuals vs fitted
ax3 = axes[1, 0]
ax3.scatter(analysis_df['ols_predicted'], analysis_df['ols_resid'], alpha=0.6)
ax3.axhline(0, color='red', linestyle='--')
ax3.set_xlabel('Fitted Values')
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals vs Fitted (check for heteroscedasticity)')

# Label high leverage points
for _, row in analysis_df.nlargest(5, 'cooks_d').iterrows():
    ax3.annotate(row['geo_id'], (row['ols_predicted'], row['ols_resid']), fontsize=8)

# 4. Cook's distance
ax4 = axes[1, 1]
ax4.bar(range(len(analysis_df)), analysis_df.sort_values('cooks_d', ascending=False)['cooks_d'].values)
ax4.axhline(4/len(analysis_df), color='red', linestyle='--', label='4/n threshold')
ax4.set_xlabel('Country (sorted by influence)')
ax4.set_ylabel("Cook's Distance")
ax4.set_title('Influential Observations')
ax4.legend()

plt.tight_layout()
plt.savefig('/Users/ilanstrauss/anthropic-econ-critique/ols_diagnostics.png', dpi=150)
plt.close()

# ============================================================
# 8. SUMMARY TABLE
# ============================================================

summary_table = pd.DataFrame({
    'Method': ['Anthropic OLS', 'Robust (Huber)', 'Low GDP only', 'Mid GDP only', 'High GDP only'],
    'Slope (β)': [
        ols_model.params['log_gdp'],
        rlm_model.params['log_gdp'],
        tercile_results.get('Low', {}).get('slope', np.nan),
        tercile_results.get('Mid', {}).get('slope', np.nan),
        tercile_results.get('High', {}).get('slope', np.nan),
    ]
})
summary_table['Slope (β)'] = summary_table['Slope (β)'].round(4)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

analysis_df.to_csv('/Users/ilanstrauss/anthropic-econ-critique/analysis_results.csv', index=False)
