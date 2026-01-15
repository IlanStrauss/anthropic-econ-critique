"""
Create figures for Anthropic Economic Index critique blog post
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

# Load data
df = pd.read_csv("/Users/ilanstrauss/anthropic-econ-critique/analysis_results.csv")

# Create figures directory
import os
os.makedirs('figures', exist_ok=True)

# =============================================================
# FIGURE 1: Their story vs Our story - Main scatter plot
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
         'k-', linewidth=3, label=f'OLS: β = 0.69')

# Confidence band (narrow - their view)
y_pred = ols.predict(sm.add_constant(x_line))
se = 0.042 * np.sqrt(1 + (x_line - df['log_gdp'].mean())**2 / df['log_gdp'].var())
ax1.fill_between(x_line, y_pred - 1.96*se*2, y_pred + 1.96*se*2, alpha=0.2, color='steelblue')

ax1.set_xlabel('ln(GDP per capita)', fontsize=12)
ax1.set_ylabel('ln(AI Usage Index)', fontsize=12)
ax1.set_title("Anthropic's View: One Global Relationship\nβ = 0.70, CI = [0.61, 0.77]", fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)

# Right panel: Our view (separate lines by income)
ax2 = axes[1]

for tercile in ['Low', 'Mid', 'High']:
    subset = df[df['income_tercile'] == tercile]
    ax2.scatter(subset['log_gdp'], subset['log_usage'], c=colors[tercile],
                alpha=0.7, s=60, edgecolor='white', linewidth=0.5, label=f'{tercile} income')

    # Fit line for each group
    X_sub = sm.add_constant(subset['log_gdp'])
    ols_sub = sm.OLS(subset['log_usage'], X_sub).fit()
    x_sub = np.linspace(subset['log_gdp'].min(), subset['log_gdp'].max(), 50)
    ax2.plot(x_sub, ols_sub.params['const'] + ols_sub.params['log_gdp'] * x_sub,
             c=colors[tercile], linewidth=2.5, linestyle='-')

ax2.set_xlabel('ln(GDP per capita)', fontsize=12)
ax2.set_ylabel('ln(AI Usage Index)', fontsize=12)
ax2.set_title("Our View: Relationship Varies by Income Level\nSlopes: 0.76 (Low), 0.44 (Mid), 0.63 (High)", fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('figures/fig1_their_view_vs_ours.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig1_their_view_vs_ours.png")

# =============================================================
# FIGURE 2: Slope comparison by income level
# =============================================================

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Low\nIncome', 'Mid\nIncome', 'High\nIncome', 'Global\n(Anthropic)']
slopes = [0.76, 0.44, 0.63, 0.69]
errors = [0.19, 0.18, 0.20, 0.042]
bar_colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']

bars = ax.bar(categories, slopes, yerr=errors, capsize=8, color=bar_colors,
              edgecolor='black', linewidth=1.5, alpha=0.8)

ax.axhline(0.69, color='#3498db', linestyle='--', linewidth=2, alpha=0.7, label="Anthropic's global estimate")
ax.axhline(0.44, color='#f39c12', linestyle=':', linewidth=2, alpha=0.7)

ax.set_ylabel('GDP Elasticity (β)', fontsize=13)
ax.set_title('The GDP-AI Usage Relationship Varies Dramatically\nMiddle-Income Countries Show Weak Effect', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)

# Add value labels
for bar, slope, err in zip(bars, slopes, errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.03,
            f'{slope:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Annotation
ax.annotate('Middle-income countries:\nGDP barely predicts AI usage!',
            xy=(1, 0.44), xytext=(1.5, 0.2),
            fontsize=11, ha='left',
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2))

plt.tight_layout()
plt.savefig('figures/fig2_slope_by_income.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_slope_by_income.png")

# =============================================================
# FIGURE 3: Confidence interval comparison
# =============================================================

fig, ax = plt.subplots(figsize=(10, 5))

methods = ['Anthropic\n(OLS)', 'Partial\nPooling']
point_estimates = [0.69, 0.66]
ci_lower = [0.61, 0.43]
ci_upper = [0.77, 0.89]

y_pos = [1, 0]
colors = ['#3498db', '#9b59b6']

for i, (method, est, low, high, c) in enumerate(zip(methods, point_estimates, ci_lower, ci_upper, colors)):
    ax.plot([low, high], [y_pos[i], y_pos[i]], color=c, linewidth=8, solid_capstyle='round', alpha=0.6)
    ax.plot(est, y_pos[i], 'o', color=c, markersize=15, markeredgecolor='white', markeredgewidth=2)
    ax.text(high + 0.02, y_pos[i], f'[{low:.2f}, {high:.2f}]', va='center', fontsize=11)

ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlim(0.3, 1.05)
ax.set_ylim(-0.5, 1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=12)
ax.set_xlabel('GDP Elasticity (β)', fontsize=13)
ax.set_title('Anthropic Underestimates Uncertainty by ~3x\nTrue confidence interval is much wider', fontsize=14, fontweight='bold')

# Annotation
ax.annotate('Their narrow CI\ngives false precision',
            xy=(0.69, 1), xytext=(0.4, 1.3),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.annotate('True CI includes\nvalues as low as 0.43',
            xy=(0.43, 0), xytext=(0.35, -0.3),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

plt.tight_layout()
plt.savefig('figures/fig3_confidence_intervals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_confidence_intervals.png")

# =============================================================
# FIGURE 4: Outliers and influential observations
# =============================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Color by residual direction
colors = ['#e74c3c' if r > 0 else '#3498db' for r in df['ols_resid']]
sizes = 50 + 500 * df['cooks_d']  # Size by influence

scatter = ax.scatter(df['log_gdp'], df['log_usage'], c=colors, s=sizes,
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

ax.set_xlabel('ln(GDP per capita)', fontsize=12)
ax.set_ylabel('ln(AI Usage Index)', fontsize=12)
ax.set_title('Influential Outliers Drive the Results\nPoint size = influence (Cook\'s D), Red = over-predicted, Blue = under-predicted',
             fontsize=13, fontweight='bold')

# Legend
red_patch = mpatches.Patch(color='#e74c3c', label='Over-adopter (above line)')
blue_patch = mpatches.Patch(color='#3498db', label='Under-adopter (below line)')
ax.legend(handles=[red_patch, blue_patch], loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig4_outliers.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_outliers.png")

# =============================================================
# FIGURE 5: Policy implications diagram
# =============================================================

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Create text boxes
anthropic_box = dict(boxstyle='round,pad=0.5', facecolor='#3498db', alpha=0.3)
our_box = dict(boxstyle='round,pad=0.5', facecolor='#9b59b6', alpha=0.3)

# Anthropic side
ax.text(0.25, 0.85, "ANTHROPIC'S STORY", fontsize=14, fontweight='bold', ha='center',
        transform=ax.transAxes, bbox=anthropic_box)
ax.text(0.25, 0.65, "GDP → AI Adoption\n(β = 0.70)", fontsize=12, ha='center', transform=ax.transAxes)
ax.text(0.25, 0.45, "Policy: Grow GDP\nto increase AI adoption", fontsize=11, ha='center',
        transform=ax.transAxes, style='italic')
ax.text(0.25, 0.25, "✓ Low-income: Correct\n✗ Mid-income: WRONG\n~ High-income: Varies",
        fontsize=11, ha='center', transform=ax.transAxes)

# Our side
ax.text(0.75, 0.85, "OUR FINDINGS", fontsize=14, fontweight='bold', ha='center',
        transform=ax.transAxes, bbox=our_box)
ax.text(0.75, 0.65, "GDP → AI varies by context\n(β = 0.44 to 0.76)", fontsize=12, ha='center', transform=ax.transAxes)
ax.text(0.75, 0.45, "Policy: Context-specific\ninterventions needed", fontsize=11, ha='center',
        transform=ax.transAxes, style='italic')
ax.text(0.75, 0.25, "Low: Economic development first\nMid: Education, infrastructure\nHigh: Cultural/policy factors",
        fontsize=11, ha='center', transform=ax.transAxes)

# Arrow
ax.annotate('', xy=(0.55, 0.5), xytext=(0.45, 0.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='gray'),
            transform=ax.transAxes)
ax.text(0.5, 0.55, 'Same data,\ndifferent story', fontsize=10, ha='center',
        transform=ax.transAxes, style='italic', color='gray')

ax.set_title('Different Methods, Different Policy Implications', fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('figures/fig5_policy_implications.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig5_policy_implications.png")

print("\n✓ All figures saved to figures/ directory")
