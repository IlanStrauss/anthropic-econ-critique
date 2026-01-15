# Does GDP Predict AI Adoption? A Reanalysis of Anthropic's Data

**Ilan Strauss | AI Disclosures Project | January 2026**

---

Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) claims:

> "At the country level, a 1% increase in GDP per capita is associated with a 0.7% increase in Claude usage per capita."

## Why This Matters

If GDP drives AI adoption, productivity gains concentrate in wealthy nations. As Anthropic's head of economics Peter McCrory told the *Financial Times*: "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards."

The policy implication: countries must grow GDP first, then AI follows. Education, infrastructure, language access—secondary.

We reanalyzed their [public data](https://huggingface.co/datasets/Anthropic/EconomicIndex). The 0.7 elasticity doesn't hold universally.

---

## Results

### 1. The relationship varies by income level

![Figure 1](figures/fig1_their_view_vs_ours.png)

| Income Level | GDP Elasticity (β) |
|--------------|-------------------|
| Low income | 0.76 |
| **Middle income** | **0.44** |
| High income | 0.63 |
| Anthropic (pooled) | 0.70 |

Middle-income countries show a 37% weaker relationship than Anthropic's global estimate.

![Figure 2](figures/fig2_slope_by_income.png)

### 2. Uncertainty is underestimated

![Figure 3](figures/fig3_confidence_intervals.png)

| Method | Slope | SE | 95% CI |
|--------|-------|-----|--------|
| Anthropic (OLS) | 0.69 | 0.042 | [0.61, 0.77] |
| Partial pooling | 0.66 | 0.116 | [0.43, 0.89] |

Standard errors are ~3x larger when accounting for group-level variance. Their confidence interval excludes 0.44—the actual middle-income elasticity.

### 3. Outliers drive results

![Figure 4](figures/fig4_outliers.png)

Six countries exceed Cook's D threshold (4/n = 0.035):

| Country | Cook's D | Issue |
|---------|----------|-------|
| Tanzania | 0.106 | Under-adopter |
| Angola | 0.097 | Under-adopter |
| Israel | 0.042 | Over-adopter (3x predicted) |
| Qatar | 0.048 | Rich, low adoption |
| Kuwait | 0.037 | Rich, low adoption |

---

## Methods

Anthropic uses OLS on log-transformed data, pooling all countries. This assumes a constant slope globally.

We use partial pooling (mixed effects models), allowing slopes to vary by income group while shrinking toward the global mean. This approach:
- Properly accounts for group-level variance ([Gelman & Hill 2007](https://www.cambridge.org/highereducation/books/data-analysis-using-regression-and-multilevel-hierarchical-models/32A29531C7FD730C3A68951A17C9D983))
- Dominates both complete pooling and no pooling ([Stein 1956](https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator); [McElreath 2017](https://elevanth.org/blog/2017/08/24/multilevel-regression-as-default/))
- Yields appropriate uncertainty intervals

---

## Policy Implications

| Income Level | Anthropic's Implication | Our Finding |
|--------------|------------------------|-------------|
| Low | GDP growth → AI adoption | Supported (β = 0.76) |
| Middle | GDP growth → AI adoption | Weak evidence (β = 0.44) |
| High | GDP growth → AI adoption | Variable; outliers dominate |

For middle-income countries (Brazil, Mexico, Thailand, Indonesia), GDP growth alone won't drive AI adoption. Education, infrastructure, and language access likely matter more.

The "divergence in living standards" Anthropic warns of is not inevitable. It depends on policy—policy their analysis obscures by pooling heterogeneous relationships.

---

## Replication

Code and data: [github.com/IlanStrauss/anthropic-econ-critique](https://github.com/IlanStrauss/anthropic-econ-critique)

---

*Contact: ilan@aidisclosures.org*
