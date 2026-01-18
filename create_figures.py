"""
Create figures for Anthropic Economic Index critique
Updated to show temporal instability and middle-income insignificance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

# Load November 2025 analysis results
df = pd.read_csv("/Users/ilanstrauss/anthropic-econ-critique/analysis_results.csv")

# Create figures directory
import os
os.makedirs('figures', exist_ok=True)

# =============================================================
# FIGURE 1: GDP vs AI Usage by Income Group (November 2025 data)
# =============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors for income terciles
colors = {'Low': '#e74c3c', 'Mid': '#f39c12', 'High': '#27ae60'}

# Left panel: Anthropic's view (single line)
ax1 = axes[0]
ax1.scatter(df['log_gdp'], df['log_usage'], c='steelblue', alpha=0.6, s=60, edgecolor='white', linewidth=0.5)

# OLS fit
X = sm.add_constant(df['log_gdp'])
ols = sm.OLS(df['log_usage'], X).fit()
x_line = np.linspace(df['log_gdp'].min(), df['log_gdp'].max(), 100)
ax1.plot(x_line, ols.params['const'] + ols.params['log_gdp'] * x_line,
         'k-', linewidth=3, label=f'OLS: beta = {ols.params["log_gdp"]:.2f}')

ax1.set_xlabel('ln(GDP per working-age capita)', fontsize=12)
ax1.set_ylabel('ln(AI Usage Index)', fontsize=12)
ax1.set_title("Anthropic's View: One Global Relationship\nbeta = 0.71, p < 0.001", fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)

# Right panel: Our view (separate lines by income)
ax2 = axes[1]

for tercile in ['Low', 'Mid', 'High']:
    subset = df[df['income_group'] == tercile]
    ax2.scatter(subset['log_gdp'], subset['log_usage'], c=colors[tercile],
                alpha=0.7, s=60, edgecolor='white', linewidth=0.5, label=f'{tercile} income')

    # Fit line for each group
    X_sub = sm.add_constant(subset['log_gdp'])
    ols_sub = sm.OLS(subset['log_usage'], X_sub).fit()
    x_sub = np.linspace(subset['log_gdp'].min(), subset['log_gdp'].max(), 50)

    # Dashed line for non-significant (middle income)
    linestyle = '--' if tercile == 'Mid' else '-'
    ax2.plot(x_sub, ols_sub.params['const'] + ols_sub.params['log_gdp'] * x_sub,
             c=colors[tercile], linewidth=2.5, linestyle=linestyle)

ax2.set_xlabel('ln(GDP per working-age capita)', fontsize=12)
ax2.set_ylabel('ln(AI Usage Index)', fontsize=12)
ax2.set_title("Our View: Middle-Income Relationship NOT Significant\nLow: 0.85*, Mid: 0.73 (p=0.105), High: 0.67*", fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)

# Add annotation for middle income
ax2.annotate('Middle income:\nNOT significant\n(p = 0.105)',
             xy=(10.5, 0.5), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3))

plt.tight_layout()
plt.savefig('figures/fig1_their_view_vs_ours.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================
# FIGURE 2: Uncertainty by Income Group (November 2025 data)
# =============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# November 2025 data only
groups = ['Global', 'Low\nIncome', 'Middle\nIncome', 'High\nIncome']
x_pos = np.arange(len(groups))

slopes = [0.71, 0.85, 0.73, 0.67]
errors = [0.06, 0.18, 0.44, 0.16]
significant = [True, True, False, True]  # Middle income NOT significant

# Colors: highlight middle income in orange
colors = ['#3498db', '#27ae60', '#e74c3c', '#27ae60']

# Plot bars with error bars
bars = ax.bar(x_pos, slopes, width=0.6, color=colors, alpha=0.8,
              yerr=errors, capsize=8, error_kw={'linewidth': 2, 'capthick': 2})

# Add significance markers
for i, (sig, slope, err) in enumerate(zip(significant, slopes, errors)):
    if sig:
        ax.text(x_pos[i], slope + err + 0.05, '*', ha='center', fontsize=18, fontweight='bold')
    else:
        ax.text(x_pos[i], slope + err + 0.05, 'n.s.', ha='center', fontsize=11, color='red', fontweight='bold')

# Add value labels
for i, (slope, err) in enumerate(zip(slopes, errors)):
    ax.text(x_pos[i], slope/2, f'β={slope:.2f}\nSE={err:.2f}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

ax.set_ylabel('GDP Elasticity (β) with Standard Error', fontsize=13)
ax.set_title('Middle-Income Relationship is Weak and Highly Uncertain\n(November 2025 data; * = p < 0.05, n.s. = not significant)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylim(0, 1.4)

# Annotation for middle income
ax.annotate('SE is 2-3x larger\nthan other groups\n(p = 0.105)',
            xy=(2, 1.2), ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.2))

plt.tight_layout()
plt.savefig('figures/fig2_uncertainty_by_group.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================
# FIGURE 3: R-squared comparison - GDP explains little for middle income
# =============================================================

fig, ax = plt.subplots(figsize=(10, 6))

groups = ['Global', 'Low Income', 'Middle Income', 'High Income']
r2_aug = [0.71, 0.30, 0.14, 0.21]
r2_nov = [0.56, 0.37, 0.07, 0.33]

x_pos = np.arange(len(groups))
width = 0.35

bars1 = ax.bar(x_pos - width/2, r2_aug, width, label='Aug 4-11, 2025', color='#3498db', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, r2_nov, width, label='Nov 13-20, 2025', color='#e74c3c', alpha=0.8)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}',
            ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}',
            ha='center', va='bottom', fontsize=10)

ax.set_ylabel('R-squared (variance explained)', fontsize=13)
ax.set_title('GDP Explains Almost Nothing for Middle-Income Countries\nR-squared fell from 14% to just 7%',
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(groups, fontsize=11)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 0.85)

# Highlight middle income
ax.axvspan(1.5, 2.5, alpha=0.1, color='orange')

plt.tight_layout()
plt.savefig('figures/fig3_r_squared.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================
# FIGURE 4: Standard Errors - Middle income has huge uncertainty
# =============================================================

fig, ax = plt.subplots(figsize=(10, 6))

groups = ['Global', 'Low Income', 'Middle Income', 'High Income']
se_aug = [0.04, 0.19, 0.18, 0.20]
se_nov = [0.06, 0.18, 0.44, 0.16]

x_pos = np.arange(len(groups))
width = 0.35

bars1 = ax.bar(x_pos - width/2, se_aug, width, label='Aug 4-11, 2025', color='#3498db', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, se_nov, width, label='Nov 13-20, 2025', color='#e74c3c', alpha=0.8)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}',
            ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}',
            ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Standard Error', fontsize=13)
ax.set_title('Middle-Income Standard Error More Than Doubled\n(0.18 to 0.44 - relationship is essentially noise)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(groups, fontsize=11)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 0.55)

# Highlight middle income
ax.axvspan(1.5, 2.5, alpha=0.1, color='orange')

plt.tight_layout()
plt.savefig('figures/fig4_standard_errors.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================
# FIGURE 5: Outliers and influential observations (November data)
# =============================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Color by residual direction
colors_resid = ['#e74c3c' if r > 0 else '#3498db' for r in df['ols_resid']]
sizes = 50 + 500 * df['cooks_d']  # Size by influence

scatter = ax.scatter(df['log_gdp'], df['log_usage'], c=colors_resid, s=sizes,
                     alpha=0.6, edgecolor='white', linewidth=0.5)

# OLS line
x_line = np.linspace(df['log_gdp'].min(), df['log_gdp'].max(), 100)
ax.plot(x_line, ols.params['const'] + ols.params['log_gdp'] * x_line,
        'k--', linewidth=2, alpha=0.7)

# Label influential points
influential = df.nlargest(8, 'cooks_d')
for _, row in influential.iterrows():
    ax.annotate(row['geo_id'], (row['log_gdp'], row['log_usage']),
                fontsize=10, fontweight='bold',
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('ln(GDP per working-age capita)', fontsize=12)
ax.set_ylabel('ln(AI Usage Index)', fontsize=12)
ax.set_title('Outliers Highlight Factors Beyond GDP\nPoint size = influence, Red = above prediction, Blue = below',
             fontsize=13, fontweight='bold')

# Legend
red_patch = mpatches.Patch(color='#e74c3c', label='Over-adopter (above line)')
blue_patch = mpatches.Patch(color='#3498db', label='Under-adopter (below line)')
ax.legend(handles=[red_patch, blue_patch], loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig5_outliers.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figures generated successfully!")
print("- fig1_their_view_vs_ours.png")
print("- fig2_temporal_instability.png")
print("- fig3_r_squared.png")
print("- fig4_standard_errors.png")
print("- fig5_outliers.png")
