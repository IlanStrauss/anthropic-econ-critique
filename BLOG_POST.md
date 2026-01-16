# Does GDP per Capita Really Predict AI Adoption? Adjusting Anthropic's Estimates
*Why Anthropic's data does not predict divergence between countries from global AI adoption -- if anything it predicts convergence*

**Ilan Strauss | [AI Disclosures Project](https://www.ssrc.org/programs/ai-disclosures-project/) | January 2026**

---

Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) claims:

> "Worldwide, uneven adoption remains well-explained by GDP per capita."

> "At the country level, a 1% increase in GDP per capita[^2] is associated with a 0.7% increase in Claude usage per capita."

**Using their own data, however, we show this claim is not true for the middle-income group of countries, where much of the world's population resides. It also points to convergence in AI adoption, not divergence.**

*Note: Anthropic's research paper contains a number of important findings — on occupational exposure, task automation, and more. We focus here on just one claim: the GDP per capita relationship.*

## Why This Matters

The head of economics at Anthropic, Peter McCrory, told the *Financial Times*: "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards." However, Anthropic's latest research does not provide much evidence (if any) on the question of whether AI will create more or less convergence in global income levels between countries.

1) Anthropic's cross-country regression, estimating how income level (GDP per capita) impacts AI adoption, does not by itself imply growing divergence in global living standards between countries. Impacts on living standards from AI adoption cannot be estimated from this relationship. This requires estimating the **second order effect** of how AI adoption impacts a country's productivity, which Anthropic does not do here.

2) Anthropic's AI Usage Index (AUI)[^1] measures **relative usage intensity** — how countries rank against each other in their usage of Claude — not absolute adoption levels. The estimated regression coefficients from the GDP per capita-AI Adoption relationship can only tell us that income levels predicts that certain countries use Claude more *relative to other countries*, but says nothing about whether that usage is economically meaningful at all. A country could rank highly on AUI while still having negligible actual AI adoption.

3) Moreover, as we focus on in this note, Anthropic's data when analyzed properly shows more reason to predict convergence in AI adoption: since **middle-income countries are adopting AI more than what their income level predicts**, when allowing for GDP per capita's impact on AI adoption to vary by a country's initial income level. Additionally, low-income countries have the largest coefficient (0.76 vs 0.63 for high-income), meaning a given percentage increase in their GDP per capita translates into greater AI adoption gains for these poorer countries than richer ones. This is the opposite of the divergence story.[^3]

### Do not pool data: all countries are not alike
Anthropic uses GDP per capita to predict AI adoption. They "pool" across all countries, meaning they assume the relationship between how income level impacts AI adoption is the same everywhere — a single parameter represents the strength and nature of this relationship and applies equally to Nigeria and Norway alike. 

We reanalyzed Anthropic's [public data](https://huggingface.co/datasets/Anthropic/EconomicIndex). The 0.7 elasticity coefficient representing the impact that gdp per capita has on AI adoption does not hold universally across countries. Their analysis is [biased](https://rodorigo.wordpress.com/wp-content/uploads/2020/02/cheng-hsiao-analysis-of-panel-dataz-lib.org_.pdf) by assuming a single relationship when in fact different ones exist for different income-level groups (see [PARTIAL_POOLING.md](PARTIAL_POOLING.md) for full analysis).

We find the relationship breaks down mostly for middle-income countries. This matters because middle-income countries contain much of the world's population. For them, income level is a weak predictor of AI adoption — and in fact are adopting AI more than their wealth would predict.

The implication: middle-income countries like Brazil, Mexico, Thailand, and Malaysia do not need to wait for more GDP growth in order to get more AI adoption — and aren't. Selective investments in education, digital infrastructure, English proficiency, and regulatory environment may be driving greater adoption. These are actionable policy levers.

Elsewhere, Anthropic also finds that human education — the sophistication of user prompts — correlates with AI adoption. We focus on their GDP per capita claim, which drives the headline, but their education finding supports our argument: middle-income countries can invest in education, for example, rather than waiting to get richer in order to drive AI adoption.

---

## Results

### 1. The relationship varies by income level

![Figure 1](figures/fig1_their_view_vs_ours.png)

**Figure 1** shows the relationship between GDP per capita (income level) and AI usage. Left panel: Anthropic's approach — one estimated OLS regression line (slope) through all countries. Right panel: separate OLS estimated regression lines (slop coefficients) by income tercile.

**The estimated slope relationships differ substantially by country income group**. Low-income countries (red) show a steep relationship; middle-income countries (orange) show a shallow one. Anthropic's single line averages over this heterogeneity, obscuring that middle-income countries achieve AI adoption beyond what their income level alone would predict. 

*But we can see considerable heterogeneity (i.e. differences) within other income groups too*. And we run a partial pooling regression elsewhere that recognizes this ([PARTIAL_POOLING.md](PARTIAL_POOLING.md)).

**A striking example: South Korea vs USA**. The United States has a GDP per capita of $132,532 — **2.6 times** South Korea's $51,496. Yet South Korea's AI Usage Index (3.73) is actually *slightly higher* than the USA's (3.62). If GDP per capita were the primary driver of AI adoption, as Anthropic's headline implies, then the USA might be expected to have considerably higher adoption. It does not. 

This also suggests that first-mover advantage in AI innovation does not necessarily translate into higher adoption, contrary to the divergence narrative in the *Financial Times* article cited above. Education, digital infrastructure, and cultural factors clearly matter more than income alone.

**Note on data coverage and income-group definitions:** 

China is not included in Anthropic's dataset. 

India and Indonesia are included but classified as *low-income* by Anthropic, who simply take countries' GDP per working-age capita and divide all the countries in their sample into three equal groups. This is a clear classification error. Both Indonesia (upper-middle) and India (lower-middle) are middle-income economies according to the World Bank's definition. 

The 38 middle-income countries in Anthropic's sample (i.e. using their income group definition based on dividing their dataset into thirds and calling each third a distinct income group) range from South Africa ($9,273 GDP/capita) to Poland ($38,209), and includes Brazil, Mexico, Thailand, Malaysia, Colombia, Argentina, Turkey, Chile, Peru, and Romania.

| Income Level | GDP Per Capita (β coefficient) | N |
|--------------|-------------------|---|
| Low income | 0.76 | 38 |
| **Middle income** | **0.44** | **38** |
| High income | 0.63 | 38 |
| Anthropic (pooled) | 0.70 | 114 |

Middle-income countries show a 37% weaker relationship between their GDP per capita and their AI usage than Anthropic's global estimate.

<img src="figures/fig2_slope_by_income.png" alt="Figure 2" width="720">

**Figure 2** shows the regression coefficients (with standard errors) estimated from **three separate OLS regressions**—one for each income tercile. Each bar represents the coefficient on log GDP per capita from that group's regression. The middle-income coefficient (0.44) is notably smaller than Anthropic's pooled global estimate (0.70).

This matters because middle-income countries contain much of the world's population. For them, income level is a weak predictor of AI adoption — they are adopting AI beyond what their wealth would predict.

**Evidence from residuals:** We can verify this claim by examining prediction errors. Using Anthropic's own regression, middle-income countries have a median residual of +0.122 — meaning the typical middle-income country adopts about 13% more AI than predicted (e^0.122 ≈ 1.13). Nearly two-thirds (63%) of middle-income countries are above Anthropic's regression line.

| Income Group | Mean Residual | Median Residual | % Above Prediction |
|--------------|---------------|-----------------|-------------------|
| Low | -0.059 | +0.042 | 50.0% |
| **Middle** | **+0.094** | **+0.122** | **63.2%** |
| High | -0.035 | -0.054 | 44.7% |

Our income-group-specific regressions also tell this story: the middle-income regression has both a lower slope (β = 0.44) and a higher intercept (-4.43 vs -7.01 for Anthropic's global model), meaning it predicts higher baseline adoption for middle-income countries at typical GDP levels. Both models have low R² for this group (~10-14%), confirming GDP per capita is a weak predictor regardless of specification.

The implication: middle-income countries like Brazil, Mexico, Thailand, and Malaysia do not need to wait for more GDP growth in order to get more AI adoption — and aren't. Selective investments in education, digital infrastructure, English proficiency, and regulatory environment may be driving adoption. These are actionable policy levers.

### 2. Anthropic's single estimate masks real heterogeneity

Anthropic reports a single global estimate (0.69) with a narrow confidence interval [0.61, 0.77]. But this interval is misleading because it assumes the relationship is constant across all countries.

**Separate regressions by income group** show slopes range from 0.44 to 0.76:

| Income Group | Slope (β) | SE | 95% CI |
|--------------|-----------|-----|--------|
| Low-income | 0.76 | 0.19 | [0.39, 1.14] |
| **Middle-income** | **0.44** | **0.18** | **[0.09, 0.79]** |
| High-income | 0.63 | 0.20 | [0.23, 1.03] |
| Anthropic (pooled) | 0.69 | 0.04 | [0.61, 0.77] |

Anthropic's confidence interval excludes the middle-income slope (0.44) entirely — their interval reflects false precision because it ignores this heterogeneity.

**Bayesian hierarchical model** (see [PARTIAL_POOLING.md](PARTIAL_POOLING.md) for full analysis): When we properly account for group-level variance using partial pooling with 7 country groups, we estimate a global slope of **β = 0.54** with 95% CI **[0.33, 0.74]**. Anthropic's estimate of 0.69 falls outside this interval.

| Method | Slope (β) | SE | 95% CI |
|--------|-----------|-----|--------|
| Anthropic (pooled OLS) | 0.69 | 0.04 | [0.61, 0.77] |
| Bayesian hierarchical (7 groups) | 0.54 | 0.10 | [0.33, 0.74] |

The Bayesian model's lower estimate (0.54 vs 0.69) reflects a composition effect: Gulf states (high GDP, low adoption) and low-income African countries (low GDP, low adoption) create a steeper apparent slope in pooled regression than exists within any group (Simpson's paradox).

### 3. Notable outliers

Some countries deviate substantially from the income-AI adoption relationship. **Israel stands out as the most striking outlier among high-income countries**: with a GDP per capita of $90,237, its AI Usage Index of 7.00 is **3x higher than the 2.36 predicted by Anthropic's regression**. Israel is the second-largest positive outlier in the entire dataset (after Georgia at 3.3x), suggesting that factors like tech sector concentration, education, and startup culture drive AI adoption far more than income alone. Gulf states (Qatar, Kuwait, Saudi Arabia) show the opposite pattern — far less AI usage than their wealth predicts. Several African countries (Tanzania, Angola) also fall well below the regression line.

These outliers suggest country-specific factors — language, culture, regulation, tech infrastructure — matter beyond income level. However, removing outliers only shifts the slope by ~5%, so the main critique (heterogeneity by income level) stands regardless.

---

## Methods

Anthropic uses OLS on log-transformed data, pooling all countries. This assumes a constant slope globally. As [Hsiao (2022, p. 12)](https://rodorigo.wordpress.com/wp-content/uploads/2020/02/cheng-hsiao-analysis-of-panel-dataz-lib.org_.pdf) notes in *Analysis of Panel Data*, pooled regression "implicitly assumes that the average values of variables and the relationships between variables are constant over time and across all cross-sectional units"—an assumption we test and find wanting.

We run **three separate OLS regressions**—one for each income tercile (low, middle, high):

```
ln(AUI) = α + β × ln(GDP per capita) + ε
```

This reveals the heterogeneity that Anthropic's pooled estimate obscures:
- Low-income: β = 0.76, SE = 0.19
- Middle-income: β = 0.44, SE = 0.18
- High-income: β = 0.63, SE = 0.20

For a more rigorous analysis using Bayesian hierarchical models with partial pooling (which allows slopes to vary by group while shrinking toward a global mean), see [PARTIAL_POOLING.md](PARTIAL_POOLING.md).

All code and data are available in the GitHub repository.

---

## Policy Implications

| Income Level | Anthropic's Implication | Our Finding |
|--------------|------------------------|-------------|
| Low | Higher income → more AI adoption | Supported (β = 0.76) |
| Middle | Higher income → more AI adoption | Weak link (β = 0.44) — adopting beyond wealth |
| High | Higher income → more AI adoption | Variable; outliers dominate |

For middle-income countries (Brazil, Mexico, Thailand, Malaysia), income level alone doesn't determine AI adoption. Education, infrastructure, and language access are already driving adoption beyond what wealth predicts.

The "divergence in living standards" Anthropic warns of is not inevitable. It depends on policy — policy their analysis obscures by pooling heterogeneous relationships.

---

## Replication

Code and data: [github.com/IlanStrauss/anthropic-econ-critique](https://github.com/IlanStrauss/anthropic-econ-critique)

---

*Contact: ilan@aidisclosures.org*

---

[^1]: **How Anthropic measures AI adoption:** Anthropic's "AI Usage Index" (AUI) is defined as: *Country's share of Claude usage ÷ Country's share of working-age population*. An AUI of 2 means a country uses Claude at twice the rate expected given its population share. This is a measure of **relative usage intensity** — it tells you how countries rank against each other, but not whether absolute usage is economically meaningful. A country could have high AUI but still have tiny absolute adoption. The data comes from a sample of 1M Claude.ai conversations and 1M API transcripts from November 2025.

[^2]: **How Anthropic measures GDP per capita:** Anthropic uses GDP per working-age capita (ages 15-64) from the World Bank's World Development Indicators, in current US dollars (nominal). This differs from standard GDP per capita which divides by total population.

[^3]: **Caveat on convergence interpretation:** This interpretation assumes comparable GDP growth rates across income groups. If high-income countries grow faster in percentage terms, absolute gaps in AI adoption could still widen despite poorer countries having a higher coefficient. However, historically, lower-income countries have often exhibited higher GDP growth rates (conditional convergence), which would reinforce the convergence dynamic suggested here.
