# Does GDP per Capita Really Predict AI Adoption? Adjusting Anthropic's Estimates
*Why Anthropic's data does not predict divergence between countries from global AI adoption but instead convergence*

**Ilan Strauss | [AI Disclosures Project](https://www.ssrc.org/programs/ai-disclosures-project/) | January 2026**

---

Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) claims:

> "Worldwide, uneven adoption remains well-explained by GDP per capita."

> "At the country level, a 1% increase in GDP per capita[^2] is associated with a 0.7% increase in Claude usage per capita."

**Using their own data, however, we show this claim is not true for the middle-income group of countries, where much of the world's population resides. It also points to convergence in AI adoption, not divergence.** It provides no evidence on the possibility for long-term divergences in economic growth between countries from AI investments.

*Note: Anthropic's research paper contains a number of important findings — on occupational exposure, task automation, and more. We focus here on just one claim: the GDP per capita relationship.*

## Why This Matters

The head of economics at Anthropic, Peter McCrory, told the [*Financial Times*](https://www.ft.com/content/3ad44e30-c738-4356-91fb-8bb2368685c4): "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards." However, Anthropic's latest research does not provide much evidence (if any) on the question of whether AI will create more or less convergence in global income levels between countries.

1) Anthropic's cross-country regression, estimating how income level (GDP per capita) impacts AI adoption, does not by itself imply growing divergence in global living standards between countries. Impacts on *living standards* (i.e. convergence vs. divergence) from AI adoption cannot be estimated from AI adoption. This requires estimating the **second order effect** of how AI adoption impacts a country's productivity, which Anthropic does not do here.

2) Anthropic's AI Usage Index (AUI)[^1] measures **relative usage intensity** — how countries rank against each other in their usage of Claude — not absolute adoption levels. The estimated regression coefficients from the GDP per capita-AI Adoption relationship can only tell us that income levels predicts that certain countries use Claude more *relative to other countries*, but says nothing about whether that usage is economically meaningful at all. A country could rank highly on AUI while still having negligible actual AI adoption.

3) *Moreover, as we focus on in this note, Anthropic's data when analyzed properly shows more reason to predict convergence in AI adoption*: since **middle-income countries are adopting AI more than what their income level predicts**, when allowing for GDP per capita's impact on AI adoption to vary by a country's initial income level. Additionally, low-income countries have the largest coefficient (0.76 vs 0.63 for high-income), meaning a given percentage increase in their GDP per capita translates into greater AI adoption gains for these poorer countries than richer ones. This is the opposite of the divergence story.[^3]

### Do not pool data: all countries are not alike
Anthropic uses GDP per capita to predict AI adoption. They "pool" across all countries, meaning they assume the relationship between how income level impacts AI adoption is the same everywhere — a single parameter represents the strength and nature of this relationship and applies equally to Nigeria and Norway alike. 

We reanalyzed Anthropic's [public data](https://huggingface.co/datasets/Anthropic/EconomicIndex), including both the November 2025 data used in their January 2026 report and an earlier August 2025 release. The 0.7 elasticity coefficient representing the impact that GDP per capita has on AI adoption does not hold universally across countries. Their analysis is [biased](https://rodorigo.wordpress.com/wp-content/uploads/2020/02/cheng-hsiao-analysis-of-panel-dataz-lib.org_.pdf) by assuming a single relationship when in fact different ones exist for different income-level groups.

We find **the relationship breaks down for middle-income countries**. In the November 2025 data (used in their report), the relationship is not statistically significant (β = 0.73, SE = 0.44, p = 0.105); in the August 2025 data, the relationship is weak (β = 0.44, SE = 0.18, p = 0.019). This matters because middle-income countries contain much of the world's population. For them, income level is a weak and uncertain predictor of AI adoption — and in fact they are adopting AI more than their wealth would predict.

The implication: middle-income countries like Brazil, Mexico, Thailand, and Malaysia do not need to wait for more GDP growth in order to get more AI adoption — and aren't. Selective investments in education, digital infrastructure, English proficiency, and regulatory environment may be driving greater adoption. These are actionable policy levers.

Elsewhere, Anthropic also finds that human education — the sophistication of user prompts — correlates with AI adoption. We focus on their GDP per capita claim, which drives the headline, but their education finding supports our argument: middle-income countries can invest in education, for example, rather than waiting to get richer in order to drive AI adoption.

---

## Results

### 1. The relationship varies by income level

![Figure 1](figures/fig1_their_view_vs_ours.png)

**Figure 1** shows the estimated relationship between GDP per capita (income level) and AI usage from regressions. Left panel: Anthropic's regression approach — one estimated OLS regression line (slope) through all countries. Right panel: separate OLS estimated regression lines (slope coefficients) by income tercile.

**The estimated slope relationships differ substantially by country income group in our regression**. Low-income countries (red) show a steep relationship; middle-income countries (orange) show a shallow one. Anthropic's single line averages over this heterogeneity, obscuring that middle-income countries achieve AI adoption beyond what their income level alone would predict. 

*We can see considerable heterogeneity (i.e. differences) within other income groups too*. The key finding is that for middle-income countries specifically, the relationship is weak and highly uncertain.

**A striking example: South Korea vs USA**. The United States has a GDP per capita of $132,532 — **2.6 times** South Korea's $51,496. Yet South Korea's AI Usage Index (3.73) is actually *slightly higher* than the USA's (3.62). If GDP per capita were the primary driver of AI adoption, as Anthropic's headline implies, then the USA might be expected to have considerably higher adoption. It does not.

This also suggests that first-mover advantage in AI innovation does not necessarily translate into higher adoption, contrary to the divergence narrative in the [*Financial Times* article](https://www.ft.com/content/3ad44e30-c738-4356-91fb-8bb2368685c4) cited above. Education, digital infrastructure, and cultural factors clearly matter more than income alone.

| Income Group | Period | Slope (β) | Std. Error | p-value | Significant? | N |
|--------------|--------|-----------|------------|---------|--------------|---|
| **Global** | Nov 2025 | 0.71 | 0.06 | <0.001 | Yes | 116 |
| **Low income** | Nov 2025 | 0.85 | 0.18 | <0.001 | Yes | 39 |
| **Middle income** | **Nov 2025** | **0.73** | **0.44** | **0.105** | **No** | **38** |
| **High income** | Nov 2025 | 0.67 | 0.16 | <0.001 | Yes | 39 |
| Middle income | Aug 2025 | 0.44 | 0.18 | 0.019 | Yes (weak) | 38 |

**For middle-income countries, the relationship is not statistically significant** in the November 2025 data used in Anthropic's report (p = 0.105). The standard error (0.44) is 2-3x larger than other income groups, and GDP per capita explains only 7% of the variation (R² = 0.07).

<img src="figures/fig2_uncertainty_by_group.png" alt="Figure 2" width="720">

**Figure 2** shows the regression coefficients (with standard errors) estimated from **separate OLS regressions** by income group using the November 2025 data. The middle-income coefficient has a much larger error bar, reflecting high uncertainty — the relationship is statistically indistinguishable from zero.

This matters because middle-income countries contain much of the world's population. For them, income level is a weak predictor of AI adoption — they are adopting AI beyond what their wealth would predict.

**Evidence from residuals:** We can verify this claim by examining prediction errors. Using Anthropic's own regression, middle-income countries have a median residual of +0.122 — meaning the typical middle-income country adopts about 13% more AI than predicted (e^0.122 ≈ 1.13). Nearly two-thirds (63%) of middle-income countries are above Anthropic's regression line.

| Income Group | Mean Residual | Median Residual | % Above Prediction |
|--------------|---------------|-----------------|-------------------|
| Low | -0.059 | +0.042 | 50.0% |
| **Middle** | **+0.094** | **+0.122** | **63.2%** |
| High | -0.035 | -0.054 | 44.7% |

Our income-group-specific regressions also tell this story: the middle-income regression has both a lower slope (β = 0.44) and a higher intercept (-4.43 vs -7.01 for Anthropic's global model), meaning it predicts higher baseline adoption for middle-income countries at typical GDP levels. Our middle-income regression has R² = 0.14, confirming GDP per capita is a weak predictor for this group.

The implication: middle-income countries like Brazil, Mexico, Thailand, and Malaysia do not need to wait for more GDP growth in order to get more AI adoption — and aren't. Selective investments in education, digital infrastructure, English proficiency, and regulatory environment may be driving adoption. These are actionable policy levers.

### 2. The middle-income relationship is weak and highly uncertain

Anthropic reports a single global estimate (0.71) with a narrow confidence interval. But when we disaggregate by income group, **the middle-income relationship is not statistically significant** and has very high uncertainty.

| Income Group | Slope (β) | SE | p-value | R² |
|--------------|-----------|-----|---------|-----|
| Low-income | 0.85 | 0.18 | <0.001 | 0.37 |
| **Middle-income** | **0.73** | **0.44** | **0.105** | **0.07** |
| High-income | 0.67 | 0.16 | <0.001 | 0.33 |
| Global (pooled) | 0.71 | 0.06 | <0.001 | 0.56 |

Key observations:
- **Middle-income SE (0.44) is 2-3x larger** than other groups
- **Middle-income R² = 7%** — GDP explains almost nothing
- **p = 0.105** — the coefficient is statistically indistinguishable from zero

The middle-income coefficient may look similar to the global estimate (0.73 vs 0.71), but the huge standard error means we cannot conclude there is any meaningful relationship. Anthropic's pooled estimate masks this uncertainty by averaging over heterogeneous groups.

### 3. Policy Implications

| Income Level | Anthropic's Implication | Our Finding |
|--------------|------------------------|-------------|
| Low | Higher income → more AI adoption | Supported (β = 0.76) |
| Middle | Higher income → more AI adoption | Weak link (β = 0.44) — adopting beyond wealth |
| High | Higher income → more AI adoption | Variable; outliers dominate |

For middle-income countries (Brazil, Mexico, Thailand, Malaysia), income level alone does not determine AI adoption. Education, infrastructure, and language access are already driving adoption beyond what wealth predicts.

The "divergence in living standards" Anthropic warns of is not inevitable. It depends on policy — policy their analysis obscures by pooling heterogeneous relationships.

### 4. Notable outliers

Some countries deviate substantially from the income-AI adoption relationship. **Israel stands out as the most striking outlier among high-income countries**: with a GDP per capita of $90,237, its AI Usage Index of 7.00 is **3x higher than the 2.36 predicted by Anthropic's regression**. Israel is the second-largest positive outlier in the entire dataset (after Georgia at 3.3x), suggesting that factors like tech sector concentration, education, and startup culture drive AI adoption far more than income alone. Gulf states (Qatar, Kuwait, Saudi Arabia) show the opposite pattern — far less AI usage than their wealth predicts. Several African countries (Tanzania, Angola) also fall well below the regression line.

These outliers suggest country-specific factors — language, culture, regulation, tech infrastructure — matter beyond income level. However, removing outliers only shifts the slope by ~5%, so the main critique (heterogeneity by income level) stands regardless.

---

## Method

Anthropic uses OLS on log-transformed data, pooling all countries. This assumes a constant slope globally. But as [Hsiao (2022, p. 12)](https://rodorigo.wordpress.com/wp-content/uploads/2020/02/cheng-hsiao-analysis-of-panel-dataz-lib.org_.pdf) notes in *Analysis of Panel Data*, pooled regression "implicitly assumes that the average values of variables and the relationships between variables are constant over time and across all cross-sectional units" — an assumption we test and find wanting.

We run **separate OLS regressions** for each income tercile (low, middle, high):

$$\ln(\text{AUI}) = \alpha + \beta \times \ln(\text{GDP per capita}) + \varepsilon$$

Using the November 2025 data from Anthropic's January 2026 report:
- Low-income: β = 0.85 (SE = 0.18, p < 0.001, R² = 0.37)
- **Middle-income: β = 0.73 (SE = 0.44, p = 0.105, R² = 0.07)** — NOT significant
- High-income: β = 0.67 (SE = 0.16, p < 0.001, R² = 0.33)

We also compared with earlier August 2025 data, finding the middle-income relationship was weak then too (β = 0.44, SE = 0.18, p = 0.019).

**Note:** GDP per capita data is the same across both periods (2024 World Bank data). Only Claude adoption patterns changed between August and November 2025. The dramatic shift in the middle-income coefficient (0.44 → 0.73) and its standard error (0.18 → 0.44) reflects changes in adoption, not income — highlighting the instability and uncertainty of this relationship.

All code and data are available in the GitHub repository.

---

## Note on Data Coverage and Income-Group Definitions

China is not included in Anthropic's dataset.

India and Indonesia are included but classified as *low-income* by Anthropic, who simply take countries' GDP per working-age capita and divide all the countries in their sample into three equal groups. This is a clear classification error. Both Indonesia (upper-middle) and India (lower-middle) are middle-income economies according to the World Bank's definition.

The 38 middle-income countries in Anthropic's sample (i.e. using their income group definition based on dividing their dataset into thirds and calling each third a distinct income group) range from South Africa ($9,273 GDP/capita) to Poland ($38,209), and include Brazil, Mexico, Thailand, Malaysia, Colombia, Argentina, Turkey, Chile, Peru, and Romania.

---

## Replication

Code and data: [github.com/IlanStrauss/anthropic-econ-critique](https://github.com/IlanStrauss/anthropic-econ-critique)

---

*Contact: ilan@aidisclosures.org*

---

[^1]: **How Anthropic measures AI adoption:** Anthropic's "AI Usage Index" (AUI) is defined as: *Country's share of Claude usage ÷ Country's share of working-age population*. An AUI of 2 means a country uses Claude at twice the rate expected given its population share. This is a measure of **relative usage intensity** — it tells you how countries rank against each other, but not whether absolute usage is economically meaningful. A country could have high AUI but still have tiny absolute adoption. The data comes from a sample of 1M Claude.ai conversations and 1M API transcripts from November 2025.

[^2]: **How Anthropic measures GDP per capita:** Anthropic uses GDP per working-age capita (ages 15-64) from the World Bank's World Development Indicators, in current US dollars (nominal). This differs from standard GDP per capita which divides by total population.

[^3]: **Caveat on convergence interpretation:** This interpretation assumes comparable GDP growth rates across income groups. If high-income countries grow faster in percentage terms, absolute gaps in AI adoption could still widen despite poorer countries having a higher coefficient. However, historically, lower-income countries have often exhibited higher GDP growth rates (conditional convergence), which would reinforce the convergence dynamic suggested here.
