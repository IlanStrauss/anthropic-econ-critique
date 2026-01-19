# Does GDP per Capita Really Predict AI Adoption? Adjusting Anthropic's Estimates
*The GDP-AI adoption relationship is in fact weak for middle-income countries when one analyzes Anthropic's own data properly*

**Ilan Strauss | [AI Disclosures Project](https://www.ssrc.org/programs/ai-disclosures-project/) | January 2026** (v.2)

---

## Summary: Anthropic's Claims vs. Our Critique

| Anthropic's Claim | Critique | Our Evidence |
|-------------------|--------------|--------------|
| ["Uneven [AI] adoption remains well-explained by GDP per capita"](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) (global β=0.71) | GDP per capita explains little of middle-income country adoption | Middle-income group: R²=7% (with Seychelles), R²=17% (without Seychelles) |
| Single global relationship applies to all countries | Different income groups have different relationships, making pooling biased (Hsiao 2022) | Estimating separate regressions for each income group show relationships (slopes) and R² (fit) vary by income group |
| AI usage concentration ["essentially unchanged"](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) (Aug–Nov 2025) | Gini captures a static snapshot; differential growth rates compound into widening gaps | Adoption growth rates show potential inequality emerging: High-income +26%, Middle-income +14%, Low-income +22% |
| ["Divergence in living standards" possible](https://www.ft.com/content/3ad44e30-c738-4356-91fb-8bb2368685c4) (McCrory to FT) | Speculative: no second-order effects estimated (AI → productivity → growth) | Cross-country data used is a snapshot from Claude front-end usage (closer to consumer) not API (closer to firm-level), and not panel (countries over time)|
| Data reflects global AI adoption (implied by FT coverage) | Data only measures Claude usage, not total AI adoption broadly | Brazil and Thailand saw *decreased* Claude usage. But could be from competitive use of alternatives (ChatGPT), not reduced AI adoption |


---

Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) claims:

> "Worldwide, uneven adoption remains well-explained by GDP per capita."

> "At the country level, a 1% increase in GDP per capita[^2] is associated with a 0.7% increase in Claude usage per capita."

**Using their own data, however, we show this claim is not true for the middle-income group of countries, where much of the world's population resides.** It also provides no evidence on the possibility for long-term divergences in economic growth between countries resulting from AI adoption or investments. Further evidence is required to consider the question of global divergences in AI or resulting from AI.

*Note: Anthropic's research paper contains a number of important findings — on occupational exposure, task automation, and more. We focus here on just one claim: the GDP per capita relationship.*

## Why This Matters

Anthropic's head of economics told the [*Financial Times*](https://www.ft.com/content/3ad44e30-c738-4356-91fb-8bb2368685c4): "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards." But their research provides little evidence for this claim:

1. **No productivity impact estimated.** Divergence in living standards requires estimating how AI adoption affects productivity — a second-order effect Anthropic does not measure.

2. **Relative rankings, not absolute adoption.** Their AI Usage Index (AUI)[^1] measures how countries rank against each other in Claude usage, not whether usage is economically meaningful. A country could rank highly while having negligible actual adoption.

3. **GDP is a weak predictor for middle-income countries.** Excluding the Seychelles outlier (likely VPN traffic), the middle-income relationship is weak: β = 0.44, R² = 0.17 — much weaker than the global estimate of 0.71.

4. **Consumer data only, Claude only.** Country-level data comes from Claude.ai (consumer), not API (enterprise). If productivity gains stem from business adoption, the geographic analysis misses where impacts would occur. Moreover, the data measures Claude usage, not AI adoption broadly — declining Claude usage in a country (e.g., Brazil, Thailand) could reflect users switching to ChatGPT or local alternatives, not reduced AI adoption overall.

**Bottom line:** Convergence or divergence claims cannot be supported by consumer-only, Claude-only, cross-sectional data that measures relative rankings rather than absolute adoption or productivity impacts.

### Pooling masks heterogeneity

Anthropic pools all countries, assuming one relationship applies equally to Nigeria and Norway. We reanalyzed their [public data](https://huggingface.co/datasets/Anthropic/EconomicIndex) and find this assumption is [biased](https://rodorigo.wordpress.com/wp-content/uploads/2020/02/cheng-hsiao-analysis-of-panel-dataz-lib.org_.pdf) — different income groups show different relationships.

**Why does pooled R² look good while group-level R²s are weak?** This reflects the difference between *between-group* and *within-group* variation. The pooled regression (R² = 0.56) captures the obvious fact that low-income countries cluster at low GDP and low AI usage, while high-income countries cluster at high GDP and high AI usage. But *within* each income group, GDP explains far less:

**Table: Regression Results — Anthropic's Global Estimate vs. Our Country-Group Estimates**

| Income Group | Slope (β) | SE | p-value | R² |
|--------------|-----------|-----|---------|-----|
| Global (pooled) | 0.71 | 0.06 | <0.001 | 0.56 |
| Low-income | 0.85 | 0.18 | <0.001 | 0.37 |
| High-income | 0.67 | 0.16 | <0.001 | 0.33 |
| **Middle-income (with Seychelles)** | **0.73** | **0.44** | **0.105** | **0.07** |
| **Middle-income (excl. Seychelles)** | **0.44** | **0.16** | **0.011** | **0.17** |

The global R² is inflated by between-group differences. Within middle-income countries — where much of the world lives — GDP per capita explains only 17% of the variation in AI adoption. This is a form of **aggregation bias** (or the **ecological fallacy**): relationships observed at the aggregate level do not hold at the disaggregated level.

**The relationship breaks down for middle-income countries.** In November 2025 data, the middle-income coefficient is not statistically significant (β = 0.73, SE = 0.44, p = 0.105). Removing the outlier of the Seychelles ahs the effects of reducing the estimated uncertainty (making it statistically significant) but now the coefficients falls to 0.44.

**The implication:** Countries like Brazil, Mexico, Thailand, and Malaysia do not need to wait for GDP growth to drive AI adoption — and aren't. Education, infrastructure, and regulatory environment may matter more than income. These are actionable policy levers.

---

## Results

### 1. The Relationship Between Income-Level (GDP per capita) and AI Adoption Varies by Country Group

![Figure 1](figures/fig1_their_view_vs_ours.png)

**Figure 1** shows separate OLS regression lines by income tercile using Anthropic's November 2025 data. Left panel: with all countries. Right panel: excluding Seychelles.

**The key finding: one data anomaly is driving the high uncertainty in the middle-income relationship.** Seychelles has a usage index 488 times higher than the next highest middle-income country — almost certainly reflecting VPN/proxy traffic routed through this offshore jurisdiction rather than genuine local adoption. With Seychelles included (left panel), the middle-income relationship appears statistically insignificant (p = 0.105) with massive uncertainty (SE = 0.44). Without Seychelles (right panel), the relationship becomes significant but **weak** — the middle-income slope (β = 0.44) is much lower than the global slope (0.70).

This highlights two issues with Anthropic's analysis: (1) their usage data needs adjustment for VPN/proxy traffic to be meaningful for cross-country comparisons, and (2) even after removing the outlier, the GDP-AI adoption relationship is much weaker for middle-income countries than their pooled estimate suggests.

Notably, the middle-income results **without Seychelles** (β = 0.44, SE = 0.16, p = 0.011) are nearly identical to an earlier August 2025 data release (β = 0.44, SE = 0.18, p = 0.019) — before Seychelles entered the dataset. The apparent "instability" between periods was entirely driven by this single VPN-related outlier.

<img src="figures/fig2_uncertainty_by_group.png" alt="Figure 2" width="720">

**Figure 2** shows the regression coefficients (with standard errors) estimated from **separate OLS regressions** by income group using the November 2025 data. The middle-income coefficient has a much larger error bar, reflecting high uncertainty — the relationship is statistically indistinguishable from zero.

### 2. Adoption *growth* is weakest for middle-income countries (Aug–Nov 2025)

[Anthropic claims](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) that "the AUI concentration across countries was essentially unchanged between our last report and this report," based on a Gini coefficient measure. However, the Gini coefficient captures a static snapshot — it can miss differential growth rates that compound into widening gaps over time.

Comparing Anthropic's two data releases (August and November 2025), we find that the *growth* in adoption was **weakest for middle-income countries** and **strongest for high-income countries**:

**Table 1: Change in AI adoption between August and November 2025 (Seychelles excluded)**

| Income Group | N | Median Aug | Median Nov | Median % Change | % Countries Increased |
|--------------|---|------------|------------|-----------------|----------------------|
| Low | 38 | 0.26 | 0.32 | +22% | 76% |
| **Mid** | **37** | **0.91** | **1.08** | **+14%** | **84%** |
| High | 38 | 2.16 | 2.63 | +26% | 89% |

High-income countries grew nearly twice as fast as middle-income countries (+26% vs +14%). If this pattern persists, concentration will *increase*, not remain "unchanged" as Anthropic suggests. The Gini may appear stable in a short window while differential growth rates set the stage for future divergence.

Middle-income countries saw the smallest median percentage increase compared to both low-income (+22%) and high-income (+26%) groups. While 84% of middle-income countries increased adoption, the magnitude of growth lagged behind.

**Important caveat:** This data measures *Claude* usage specifically, not AI adoption broadly. Six middle-income countries actually saw *decreased* Claude adoption: Armenia (-24%), Thailand (-23%), Guatemala (-15%), Brazil (-9%), Paraguay (-1%), and Turkey (-1%). Meanwhile, Peru (+71%), Kazakhstan (+64%), and Dominican Republic (+55%) grew rapidly.

This heterogeneity could reflect:
- **Competition from ChatGPT or local alternatives** — Claude usage declining in Brazil or Thailand may indicate users switching to other AI tools, not reduced AI adoption overall
- **Language barriers** — Claude's relative strength in English may disadvantage non-English markets
- **Measurement noise** — only three months of data

The pattern is suggestive but should be interpreted cautiously.

### 3. Policy Implications

| Income Level | Anthropic's Implication | Our Finding |
|--------------|------------------------|-------------|
| Low | Higher income → more AI adoption | Supported (β = 0.76) |
| Middle | Higher income → more AI adoption | Weak link (β = 0.44) — adopting beyond wealth |
| High | Higher income → more AI adoption | Variable; outliers dominate |

For middle-income countries (Brazil, Mexico, Thailand, Malaysia), income level alone does not determine AI adoption. Education, infrastructure, and language access are already driving adoption beyond what wealth predicts.

The "divergence in living standards" Anthropic warns of is not inevitable. It depends on policy — policy their analysis obscures by pooling heterogeneous relationships.

---

## Appendix

### Notable Outliers

Some countries deviate substantially from the income-AI adoption relationship. **Israel stands out as the most striking outlier among high-income countries**: with a GDP per capita of $90,237, its AI Usage Index of 7.00 is **3x higher than the 2.36 predicted by Anthropic's regression**. Israel is the second-largest positive outlier in the entire dataset (after Georgia at 3.3x), suggesting that factors like tech sector concentration, education, and startup culture drive AI adoption far more than income alone. Gulf states (Qatar, Kuwait, Saudi Arabia) show the opposite pattern — far less AI usage than their wealth predicts. Several African countries (Tanzania, Angola) also fall well below the regression line.

These outliers suggest country-specific factors — language, culture, regulation, tech infrastructure — matter beyond income level. However, removing outliers only shifts the slope by ~5%, so the main critique (heterogeneity by income level) stands regardless.

### Evidence from Residuals

The apparent "over-adoption" by middle-income countries is driven entirely by the Seychelles outlier. With Seychelles included, 63% of middle-income countries appear above the regression line; without it, only 41% are above.

| Income Group | Mean Residual | Median Residual | % Above Prediction |
|--------------|---------------|-----------------|-------------------|
| Low | -0.059 | +0.042 | 50.0% |
| High | -0.035 | -0.054 | 44.7% |

| Middle Income Group | With Seychelles | Without Seychelles |
|---------------------|-----------------|-------------------|
| Mean Residual | **+0.094** | -0.030 |
| Median Residual | **+0.122** | -0.063 |
| % Above Prediction | **63.2%** | 40.5% |

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

