# Does GDP per Capita Really Predict AI Adoption? Adjusting Anthropic's Estimates

**Ilan Strauss | AI Disclosures Project | January 2026**

---

Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) claims:

> "Worldwide, uneven adoption remains well-explained by GDP per capita."

> "At the country level, a 1% increase in GDP per capita is associated with a 0.7% increase in Claude usage per capita."

**Using their own data, we show this is not true for middle-income countries—where most of the world's population lives.**

## Why This Matters

Anthropic's divergence scenario is less clear-cut than they suggest. The head of economics at Anthropic, Peter McCrory, told the *Financial Times*: "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards."

But a cross-sectional regression looking at how income (GDP per capita) impacts AI adoption doesn't imply divergence in living standards. Impacts on living standards from AI adoption cannot be estimated. This requires estimating how AI adoption impacts productivity, which Anthropic cannot and does not do.

Moreover, as we show below, our findings suggest more reason to predict convergence in AI adoption at least: middle-income countries are already adopting AI beyond what their income predicts, when allowing for how GDP per capita impacts AI adoption to vary by a country's starting income level.

Anthropic uses GDP per capita to predict AI adoption. They "pool" across all countries, meaning they assume the relationship between income level and AI adoption is the same everywhere—a single parameter applies to Nigeria and Norway alike.

Anthropic also finds that human education—the sophistication of user prompts—correlates with AI adoption. We focus on their GDP per capita claim, which drives the headline, but their education finding supports our argument: middle-income countries can invest in education rather than waiting to get richer to drive AI adoption.

We reanalyzed their [public data](https://huggingface.co/datasets/Anthropic/EconomicIndex). The 0.7 elasticity doesn't hold universally.

We find the relationship breaks down mostly for middle-income countries. This matters because middle-income countries contain most of the world's population. For them, income level is a weak predictor of AI adoption—they're adopting AI beyond what their wealth would predict.

The implication: middle-income countries like Brazil, Mexico, Thailand, and Indonesia don't need to wait to get richer to see more AI adoption—and aren't. Selective investments in education, digital infrastructure, English proficiency, and regulatory environment may be driving adoption. These are actionable policy levers.

---

## Results

### 1. The relationship varies by income level

![Figure 1](figures/fig1_their_view_vs_ours.png)

**Figure 1** shows the relationship between GDP per capita (income level) and AI usage. Left panel: Anthropic's approach—one regression line through all countries. Right panel: separate lines by income tercile.

The slopes differ substantially. Low-income countries (red) show a steep relationship; middle-income countries (orange) show a shallow one. Anthropic's single line averages over this heterogeneity, obscuring that middle-income countries achieve AI adoption beyond what their income level alone would predict.

| Income Level | GDP Elasticity (β) |
|--------------|-------------------|
| Low income | 0.76 |
| **Middle income** | **0.44** |
| High income | 0.63 |
| Anthropic (pooled) | 0.70 |

Middle-income countries show a 37% weaker relationship than Anthropic's global estimate.

![Figure 2](figures/fig2_slope_by_income.png)

**Figure 2** shows the estimated elasticity (with standard errors) for each income group. The middle-income bar is notably shorter than Anthropic's global estimate.

This matters because middle-income countries contain most of the world's population. For them, income level is a weak predictor of AI adoption—they're adopting AI beyond what their wealth would predict.

The implication: middle-income countries like Brazil, Mexico, Thailand, and Indonesia don't need to wait to get richer to see more AI adoption—and aren't. Selective investments in education, digital infrastructure, English proficiency, and regulatory environment may be driving adoption. These are actionable policy levers.

### 2. Uncertainty is underestimated

![Figure 3](figures/fig3_confidence_intervals.png)

**Figure 3** compares confidence intervals. The top bar (Anthropic's OLS) shows a narrow interval [0.61, 0.77]. The bottom bar (partial pooling) shows a wider interval [0.43, 0.89].

Anthropic's narrow interval suggests false precision. It excludes 0.44—the elasticity we actually observe in middle-income countries. Proper accounting for group-level variance reveals we are far less certain about the income-AI adoption relationship than Anthropic implies.

| Method | Slope | SE | 95% CI |
|--------|-------|-----|--------|
| Anthropic (OLS) | 0.69 | 0.042 | [0.61, 0.77] |
| Partial pooling | 0.66 | 0.116 | [0.43, 0.89] |

Standard errors are ~3x larger when accounting for group-level variance (i.e., variation across income groups: low, middle, high).

### 3. Notable outliers

Some countries deviate substantially from the income-AI adoption relationship. Israel has 3x the AI usage predicted by its income level. Gulf states (Qatar, Kuwait, Saudi Arabia) have far less AI usage than their wealth predicts. Several African countries (Tanzania, Angola) also fall well below the regression line.

These outliers suggest country-specific factors—language, culture, regulation, tech infrastructure—matter beyond income level. However, removing outliers only shifts the slope by ~5%, so the main critique (heterogeneity by income level) stands regardless. See [CRITIQUE.md](CRITIQUE.md) for detailed outlier analysis.

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
| Low | Higher income → more AI adoption | Supported (β = 0.76) |
| Middle | Higher income → more AI adoption | Weak link (β = 0.44) — adopting beyond wealth |
| High | Higher income → more AI adoption | Variable; outliers dominate |

For middle-income countries (Brazil, Mexico, Thailand, Indonesia), income level alone doesn't determine AI adoption. Education, infrastructure, and language access are already driving adoption beyond what wealth predicts.

The "divergence in living standards" Anthropic warns of is not inevitable. It depends on policy—policy their analysis obscures by pooling heterogeneous relationships.

---

## Replication

Code and data: [github.com/IlanStrauss/anthropic-econ-critique](https://github.com/IlanStrauss/anthropic-econ-critique)

---

*Contact: ilan@aidisclosures.org*
