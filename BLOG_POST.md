# Does GDP per Capita Really Predict AI Adoption? Adjusting Anthropic's Estimates

**Ilan Strauss | [AI Disclosures Project](https://www.ssrc.org/programs/ai-disclosures-project/) | January 2026**

---

Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) claims:

> "Worldwide, uneven adoption remains well-explained by GDP per capita."

> "At the country level, a 1% increase in GDP per capita[^2] is associated with a 0.7% increase in Claude usage per capita."

**Using their own data, we show this is not true for middle-income countries — where much of the world's population resides.**

*Note: Anthropic's research paper contains a number of important findings — on occupational exposure, task automation, and more. We focus here on just one claim: the GDP per capita relationship.*

## Why This Matters

The head of economics at Anthropic, Peter McCrory, told the *Financial Times*: "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards." Yet I try to show that Anthropic's latest research does not provide much evidence on the question of if AI will create more or less convergence in global income levels between countries.

Anthropic's cross-country regression, estimating how income (GDP per capita) impacts AI adoption, does not by itself, however, imply divergence in living standards. Impacts on living standards from AI adoption cannot be estimated from this regression. This requires estimating the **second order effect** of how AI adoption impacts productivity, which Anthropic does not do here.

Furthermore, Anthropic's AI Usage Index (AUI)[^1] measures **relative usage intensity** — how countries rank against each other — not absolute adoption levels. A 0.7 elasticity tells us richer countries use Claude more *relative to other countries*, but says nothing about whether that usage is economically meaningful in absolute terms. A country could rank highly on AUI while still having negligible actual AI adoption.

Moreover, as we focus on below, our findings suggest more reason to predict convergence in AI adoption at least: **middle-income countries are already adopting AI beyond what their income predicts**, when allowing for how GDP per capita impacts AI adoption to vary by a country's starting income level. Additionally, low-income countries have the largest coefficient (0.76 vs 0.63 for high-income), meaning a given percentage increase in GDP per capita translates into greater AI adoption gains for poorer countries than richer ones — the opposite of the divergence story.[^3]

Anthropic uses GDP per capita to predict AI adoption. They "pool" across all countries, meaning they assume the relationship between income level and AI adoption is the same everywhere — a single parameter applies to Nigeria and Norway alike.

We reanalyzed their [public data](https://huggingface.co/datasets/Anthropic/EconomicIndex). The 0.7 elasticity doesn't hold universally across countries.

We find the relationship breaks down mostly for middle-income countries. This matters because middle-income countries contain much of the world's population. For them, income level is a weak predictor of AI adoption — they are adopting AI beyond what their wealth would predict.

The implication: middle-income countries like Brazil, Mexico, Thailand, and Malaysia do not need to wait for more GDP growth in order to get more AI adoption — and aren't. Selective investments in education, digital infrastructure, English proficiency, and regulatory environment may be driving adoption. These are actionable policy levers.

Anthropic also finds that human education — the sophistication of user prompts — correlates with AI adoption. We focus on their GDP per capita claim, which drives the headline, but their education finding supports our argument: middle-income countries can invest in education rather than waiting to get richer to drive AI adoption.

---

## Results

### 1. The relationship varies by income level

![Figure 1](figures/fig1_their_view_vs_ours.png)

**Figure 1** shows the relationship between GDP per capita (income level) and AI usage. Left panel: Anthropic's approach — one regression line through all countries. Right panel: separate lines by income tercile.

The slopes differ substantially. Low-income countries (red) show a steep relationship; middle-income countries (orange) show a shallow one. Anthropic's single line averages over this heterogeneity, obscuring that middle-income countries achieve AI adoption beyond what their income level alone would predict.

**A striking example: South Korea vs USA.** The United States has a GDP per capita of $132,532 — **2.6 times** South Korea's $51,496. Yet South Korea's AI Usage Index (3.73) is actually *slightly higher* than the USA's (3.62). If GDP per capita were the primary driver of AI adoption, as Anthropic's headline implies, then the USA might be expected to have considerably higher adoption. It does not. This also suggests that first-mover advantage in AI innovation does not necessarily translate into higher adoption, contrary to the divergence narrative in the *Financial Times* article cited above. Education, digital infrastructure, and cultural factors clearly matter more than income alone.

**Note on data coverage:** China is not included in Anthropic's dataset. India and Indonesia are included but classified as *low-income* based on GDP per working-age capita — they are not in the middle-income tercile. The 38 middle-income countries range from South Africa ($9,273 GDP/capita) to Poland ($38,209), and include Brazil, Mexico, Thailand, Malaysia, Colombia, Argentina, Turkey, Chile, Peru, and Romania.

| Income Level | GDP Per Capita (β coefficient) | N |
|--------------|-------------------|---|
| Low income | 0.76 | 38 |
| **Middle income** | **0.44** | **38** |
| High income | 0.63 | 38 |
| Anthropic (pooled) | 0.70 | 114 |

Middle-income countries show a 37% weaker relationship than Anthropic's global estimate.

<img src="figures/fig2_slope_by_income.png" alt="Figure 2" width="720">

**Figure 2** shows the regression coefficients (with standard errors) estimated from a single regression with income-group-varying slopes (jointly estimated). Each bar represents the coefficient on log GDP per capita for that income group. The middle-income coefficient (0.44) is notably smaller than Anthropic's pooled global estimate (0.70). See [CRITIQUE.md](CRITIQUE.md) for the regression equation from which these coefficient estimates are derived.

This matters because middle-income countries contain much of the world's population. For them, income level is a weak predictor of AI adoption — they are adopting AI beyond what their wealth would predict.

**Evidence from residuals:** We can verify this claim by examining prediction errors. Using Anthropic's own regression, middle-income countries have a median residual of +0.122 — meaning the typical middle-income country adopts about 13% more AI than predicted (e^0.122 ≈ 1.13). Nearly two-thirds (63%) of middle-income countries are above Anthropic's regression line.

| Income Group | Mean Residual | Median Residual | % Above Prediction |
|--------------|---------------|-----------------|-------------------|
| Low | -0.059 | +0.042 | 50.0% |
| **Middle** | **+0.094** | **+0.122** | **63.2%** |
| High | -0.035 | -0.054 | 44.7% |

Our middle-income-specific regression (β = 0.44) also tells this story: it has a higher intercept than Anthropic's global model (-4.43 vs -7.01), meaning it predicts higher baseline adoption for middle-income countries. Both models have low R² for this group (~10-14%), confirming GDP per capita is a weak predictor regardless of specification. See [CRITIQUE.md](CRITIQUE.md) for full details.

The implication: middle-income countries like Brazil, Mexico, Thailand, and Malaysia do not need to wait for more GDP growth in order to get more AI adoption — and aren't. Selective investments in education, digital infrastructure, English proficiency, and regulatory environment may be driving adoption. These are actionable policy levers.

### 2. Uncertainty is underestimated

<img src="figures/fig3_confidence_intervals.png" alt="Figure 3" width="720">

**Figure 3** compares confidence intervals. The top bar (Anthropic's OLS) shows a narrow interval [0.61, 0.77]. The bottom bar (partial pooling) shows a wider interval [0.43, 0.89].

Anthropic's narrow interval suggests false precision. It excludes 0.44 — the elasticity we actually observe in middle-income countries. Proper accounting for group-level variance reveals we are far less certain about the income-AI adoption relationship than Anthropic implies.

| Method | Slope | SE | 95% CI |
|--------|-------|-----|--------|
| Anthropic (OLS) | 0.69 | 0.042 | [0.61, 0.77] |
| Partial pooling | 0.66 | 0.116 | [0.43, 0.89] |

Standard errors are ~3x larger when accounting for group-level variance (i.e., variation across income groups: low, middle, high).

**Why is the SE so much larger?** Anthropic's OLS standard error captures only sampling variance — uncertainty about the slope assuming one true global slope exists. But the slopes genuinely differ across income groups (0.44 to 0.76). Partial pooling captures both sampling variance *and* this between-group heterogeneity. OLS treats slope variation as noise around a "true" single slope; partial pooling recognizes there is no single slope. This is the classic "Moulton problem" in econometrics: regressing outcomes on group-level predictors while ignoring group structure biases standard errors downward by 2-3x ([Bertrand, Duflo & Mullainathan 2004](https://academic.oup.com/qje/article/119/1/249/1876068)).

### 3. Notable outliers

Some countries deviate substantially from the income-AI adoption relationship. **Israel stands out as the most striking outlier among high-income countries**: with a GDP per capita of $90,237, its AI Usage Index of 7.00 is **3x higher than the 2.36 predicted by Anthropic's regression**. Israel is the second-largest positive outlier in the entire dataset (after Georgia at 3.3x), suggesting that factors like tech sector concentration, education, and startup culture drive AI adoption far more than income alone. Gulf states (Qatar, Kuwait, Saudi Arabia) show the opposite pattern — far less AI usage than their wealth predicts. Several African countries (Tanzania, Angola) also fall well below the regression line.

These outliers suggest country-specific factors — language, culture, regulation, tech infrastructure — matter beyond income level. However, removing outliers only shifts the slope by ~5%, so the main critique (heterogeneity by income level) stands regardless. See [CRITIQUE.md](CRITIQUE.md) for detailed outlier analysis.

---

## Methods

Anthropic uses OLS on log-transformed data, pooling all countries. This assumes a constant slope globally.

We use partial pooling (mixed effects models) — a class of [James-Stein estimators](https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator) — allowing slopes to vary by income group while shrinking toward the global mean. This approach:
- Properly accounts for group-level variance ([Gelman & Hill 2007](https://www.cambridge.org/highereducation/books/data-analysis-using-regression-and-multilevel-hierarchical-models/32A29531C7FD730C3A68951A17C9D983); [Hsiao 2022, p. 12](https://www.cambridge.org/core/books/analysis-of-panel-data/B8C2B0B64BEB1682A845C5F3FF677E61))
- Dominates both complete pooling and no pooling ([McElreath 2017](https://elevanth.org/blog/2017/08/24/multilevel-regression-as-default/))
- Yields appropriate uncertainty intervals

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
