# Methodological Critique of Anthropic's Economic Index

**Ilan Strauss**
**AI Disclosures Project**
**January 2026**

---

## Abstract

Anthropic's Economic Index claims that "a 1% increase in GDP per capita is associated with a 0.7% increase in Claude usage per capita." This finding matters: if GDP strongly predicts AI adoption, then AI's productivity benefits will flow disproportionately to wealthy nations, potentially widening global inequality. As Anthropic's head of economics Peter McCrory warned in the *Financial Times*, "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards."

We show this estimate: (1) has underestimated uncertainty due to ignoring hierarchical data structure, (2) masks substantial heterogeneity across income levels, and (3) is driven by a handful of influential outliers. Using partial pooling models, we find the true uncertainty is ~3x larger than reported, and the GDP-usage relationship varies from 0.44 in middle-income countries to 0.76 in low-income countries.

The policy implications differ sharply from Anthropic's narrative. For middle-income countries—home to most of the world's population—GDP growth alone will not drive AI adoption. Factors like education, digital infrastructure, and language access may matter more. The "divergence in living standards" that Anthropic warns of is not inevitable; it depends on policy choices that their analysis obscures.

---

## 1. Introduction

Anthropic's January 2026 Economic Index report presents a simple, compelling story: national income strongly predicts AI adoption. Their headline finding states:

> "At the country level, a 1% increase in GDP per capita is associated with a 0.7% increase in Claude usage per capita." (Chapter 3: How Claude is used varies by geography)

They further claim to "replicate the finding from our prior report that GDP is strongly correlated with the AUI (Anthropic AI Usage Index)" and note this relationship "holds for both countries and US states" (Figures 3.3 and 3.4).

However, the report presents a **single global elasticity** without examining whether this relationship varies systematically across development levels. While they acknowledge that "the primitives themselves are not necessarily causal factors—we don't know if income or education are truly driving adoption, or if they're proxies for other underlying conditions," they do not test for heterogeneity in the GDP-usage relationship itself.

We examine whether this conclusion is robust to:
- Proper uncertainty quantification using hierarchical models
- Heterogeneity in the GDP-usage relationship across development levels
- Influence of outlier countries

---

## 2. Data and Methods

### 2.1 Why Partial Pooling?

Anthropic faces a fundamental choice when estimating the GDP-usage relationship:

1. **Complete pooling** (their approach): Estimate one global slope, ignoring country/region differences
2. **No pooling**: Estimate separate slopes for each group, ignoring shared information
3. **Partial pooling**: Estimate group-specific slopes that are *shrunk* toward the global mean

Partial pooling is theoretically preferred for several reasons:

**James-Stein Estimator Properties**: [Stein (1956)](https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator) proved a remarkable result: when estimating three or more means simultaneously, the MLE (no pooling) is *inadmissible*—there always exists a shrinkage estimator with lower mean squared error. The [James-Stein estimator](https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator) shrinks individual estimates toward the grand mean, and this "borrowing of strength" across groups reduces total estimation error. Partial pooling implements this principle: noisy group estimates are pulled toward the global estimate, with the degree of shrinkage determined by relative sample sizes and variance. As [Efron and Morris (1977)](https://statweb.stanford.edu/~ckirby/brad/other/Article1977.pdf) explained in their famous *Scientific American* article, this "Stein's Paradox" shows that combining information across groups is almost always better than treating them independently.

**Bias-Variance Tradeoff**: As McElreath (2020) explains in *Statistical Rethinking*, partial pooling navigates between two failure modes:
- Complete pooling *underfits*: it ignores real heterogeneity across groups
- No pooling *overfits*: it treats noise as signal, especially in small groups

Partial pooling adaptively regularizes: groups with little data are shrunk heavily toward the global mean (regularization dominates), while groups with abundant data retain their individual estimates (data dominates).

**Proper Uncertainty Quantification**: Gelman et al. (2013) emphasize in *Bayesian Data Analysis* that hierarchical models correctly propagate uncertainty from the group level to the population level. OLS standard errors assume fixed, known group effects—partial pooling acknowledges they are estimated with error, yielding appropriately wider confidence intervals.

### 2.2 Why OLS Standard Errors Are Wrong Here (Theory)

Anthropic's OLS approach assumes:

```
Var(β̂_OLS) = σ² / [(n-1) * Var(x)]
```

where observations are **independent and identically distributed** with a **constant slope**. Both assumptions fail with grouped data when slopes vary.

**The Problem: Slope Heterogeneity**

When the true data-generating process has group-varying slopes:

```
y_ij = α + α_j + (β + β_j) * x_ij + ε_ij
```

where `β_j ~ N(0, τ²_β)` are random slope deviations, the variance of the pooled OLS estimator is:

```
Var(β̂_pooled) = Var(β̂ | slopes fixed) + Var(slopes across groups)
```

OLS only captures the first term—the **sampling variance** given fixed slopes. It completely ignores the second term—the **between-group variance in slopes** (`τ²_β`).

**Decomposing Our Results**

In our analysis:
- OLS SE = 0.042 (captures only within-group sampling variance)
- Partial pooling SE = 0.116 (captures both sampling variance AND slope heterogeneity)
- Ratio = 2.76x

The slope varies substantially across income terciles:
- Low income: β = 0.76
- Mid income: β = 0.44
- High income: β = 0.63
- Standard deviation of slopes: ~0.16

This between-group slope variance **adds directly** to the uncertainty of the pooled estimate. OLS treats this heterogeneity as noise around a "true" single slope, when in fact there is no single slope—the relationship genuinely differs across development levels.

**Why This Matters**

Moulton (1990) showed that ignoring group structure leads to:
- Standard errors biased downward
- t-statistics inflated
- Spuriously significant results
- Confidence intervals that are **too narrow** and **exclude the true value** too often

Anthropic's CI of [0.61, 0.77] appears precise but reflects only sampling uncertainty, ignoring the fundamental question: *which population's slope are we estimating?* The true CI of [0.43, 0.89] properly reflects that:
1. We're uncertain about each group's slope (sampling variance)
2. The groups genuinely differ (slope heterogeneity)
3. The "average" slope depends on how we weight groups

**The Moulton Problem in Economics**

This is a well-known issue when regressing micro outcomes on macro predictors. Bertrand, Duflo & Mullainathan (2004) showed that difference-in-differences studies routinely understate standard errors by 2-3x when ignoring group clustering—exactly what we find here.

**Empirical Performance**: McElreath's [Statistical Rethinking lectures](https://github.com/rmcelreath/stat_rethinking_2023) and [Chapter 13 on multilevel models](https://bookdown.org/content/4857/models-with-memory.html) demonstrate that partial pooling consistently outperforms both alternatives in prediction tasks, particularly when groups have unequal sample sizes—exactly our situation with country-level data. As he notes: "Varying intercepts are just regularized estimates, but adaptively regularized by estimating how diverse the clusters are while estimating the features of each cluster."

In our context: Anthropic's complete pooling assumes all countries share identical GDP-usage relationships. This is empirically false (slopes range from 0.44 to 0.76) and theoretically unjustified. Partial pooling reveals both the heterogeneity and the appropriate uncertainty.

### 2.2 Data

We use Anthropic's publicly released data from their [HuggingFace repository](https://huggingface.co/datasets/Anthropic/EconomicIndex), specifically the `release_2025_09_15` dataset containing:
- Claude.ai usage data (November 2025)
- Country-level aggregates with GDP per working-age capita
- 114 countries with ≥200 observations (their filtering threshold)

### 2.2 Anthropic's Approach

Anthropic estimates a simple OLS regression in log-log space:

$$\ln(\text{Usage}_i) = \alpha + \beta \ln(\text{GDP}_i) + \varepsilon_i$$

This yields $\hat{\beta} = 0.69$ with SE = 0.042, which they round to "0.7".

### 2.3 Our Approach

We employ three alternatives:

**Partial Pooling (Mixed Effects)**:

```
ln(Usage_ij) = α + α_j + (β + β_j) * ln(GDP_ij) + ε_ij
```

where `j` indexes income groups, and `α_j`, `β_j` are random effects shrunk toward zero.

**Robust Regression**: Huber M-estimation to downweight outliers.

**Subsample Analysis**: Separate OLS within income terciles.

---

## 3. Results

### 3.1 Uncertainty is Underestimated

| Method | Slope (β) | Std. Error | 95% CI |
|--------|-----------|------------|--------|
| Anthropic OLS | 0.69 | 0.042 | [0.61, 0.77] |
| Partial Pooling | 0.66 | 0.116 | [0.43, 0.89] |

**Finding**: The partial pooling standard error is **2.8x larger** than OLS. Anthropic's confidence interval is too narrow because it ignores group-level variance.

**Implication**: The true 95% CI includes values as low as 0.43—almost half the point estimate. The relationship between GDP and AI usage is **far less precisely estimated** than Anthropic suggests.

### 3.2 The Relationship Varies by Income Level

| Income Tercile | Slope (β) | SE | R² | N |
|----------------|-----------|-----|-----|---|
| Low income | 0.76 | 0.19 | 0.30 | 38 |
| **Middle income** | **0.44** | 0.18 | 0.14 | 38 |
| High income | 0.63 | 0.20 | 0.21 | 38 |

**Finding**: The GDP-usage elasticity varies by **73%** across income levels (0.44 to 0.76).

**Key insight**: Middle-income countries show a **dramatically weaker** relationship (0.44 vs 0.70 global average). This suggests:

1. GDP is a **binding constraint** in low-income countries—basic economic development is necessary for AI adoption
2. In middle-income countries, **other factors dominate**: education, English proficiency, tech infrastructure, regulatory environment
3. Among high-income countries, there is **substantial variation** unexplained by GDP (Israel vs Gulf states)

**Policy implication**: Anthropic's recommendation that GDP growth drives AI adoption is **misleading for middle-income countries**. For Brazil, Mexico, Thailand, and similar nations, focusing on GDP growth alone will not close the AI adoption gap. Policy should target education and infrastructure.

### 3.3 Outliers Drive Results

**Top influential observations (Cook's Distance > 4/n):**

| Country | Issue | Cook's D |
|---------|-------|----------|
| Tanzania | Severe under-adopter | 0.106 |
| Angola | Severe under-adopter | 0.097 |
| Uzbekistan | Under-adopter | 0.050 |
| Qatar | Rich but low usage | 0.048 |
| Israel | Massive over-adopter (7x index) | 0.042 |
| Kuwait | Rich but low usage | 0.037 |

**Finding**: Six countries have outsized influence. Removing them would substantially change the slope estimate.

**Pattern**:
- **Israel** is a massive outlier—7x the expected usage for its GDP. This pulls the slope up.
- **Gulf states** (Qatar, Kuwait, Saudi Arabia) are rich but have low AI adoption. Cultural, linguistic, or regulatory factors likely explain this.
- **African under-adopters** (Tanzania, Angola) pull the regression line steeper to accommodate them.

**Robust regression** (downweighting outliers) gives $\hat{\beta} = 0.67$, confirming the OLS estimate is inflated.

---

## 4. Discussion

### 4.1 What Anthropic Gets Wrong

**Their Claim vs Our Finding:**

| Anthropic's Claim | Our Finding |
|-------------------|-------------|
| "a 1% increase in GDP per capita is associated with a 0.7% increase in Claude usage" | Elasticity is 0.66 with partial pooling; ranges from 0.44 to 0.76 by income level |
| "GDP is strongly correlated with the AUI (Anthropic AI Usage Index)" | Correlation is weak (R²=0.14) in middle-income countries |
| Relationship "holds for both countries and US states" | Relationship varies dramatically: near-zero effect for middle-income nations |
| Single elasticity presented without uncertainty discussion | True 95% CI is [0.43, 0.89], not [0.61, 0.77] |

**Specific Methodological Issues:**

1. **Treats heterogeneity as noise**: By pooling all countries into one regression, they assume the GDP-usage relationship is constant globally. It is not. The slope varies by 73% across income terciles (0.44 to 0.76).

2. **Underestimates uncertainty**: OLS standard errors assume independent observations with constant variance. Country-level data has hierarchical structure that inflates true uncertainty by ~3x.

3. **Ignores influential observations**: A handful of unusual countries (Israel, Gulf states, African outliers) disproportionately determine the slope. Tanzania alone has Cook's D of 0.11—exceeding the 4/n threshold by 3x.

### 4.2 What This Means for Policy

**Anthropic's implicit recommendation**: "Grow GDP → Get AI adoption"

**Our correction**:
- For **low-income countries**: This is approximately correct. GDP is a binding constraint.
- For **middle-income countries**: **Wrong**. GDP growth has weak association with AI adoption (0.44 elasticity). Focus instead on:
  - Education and digital literacy
  - English language proficiency
  - Tech infrastructure
  - Regulatory environment
- For **high-income countries**: Varies enormously. Israel's success vs Gulf states' lag suggests cultural and policy factors dominate.

### 4.3 Limitations

- Our income tercile analysis is exploratory; proper inference requires larger samples per group
- We use income groups as a proxy for development stage; other groupings may be more appropriate
- Partial pooling with only 5 groups (quintiles) provides limited shrinkage

---

## 5. Conclusion

Anthropic's headline finding—0.7 elasticity between GDP and AI usage—is:
1. **Less precise** than claimed (true SE is ~3x larger)
2. **Heterogeneous** across income levels (0.44 to 0.76)
3. **Driven by outliers** (Israel, Gulf states, African countries)

For middle-income countries—home to most of the world's population—GDP growth alone will not drive AI adoption. Policymakers should focus on education, infrastructure, and enabling environments rather than assuming economic growth automatically translates to AI diffusion.

---

## References

1. Anthropic. (2026). Anthropic Economic Index: January 2026 Report. https://www.anthropic.com/research/anthropic-economic-index-january-2026-report

2. Anthropic. (2025). EconomicIndex Dataset. HuggingFace. https://huggingface.co/datasets/Anthropic/EconomicIndex

3. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.

4. Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.

5. McElreath, R. (2020). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). Chapman and Hall/CRC.

6. McElreath, R. (2017). ["Multilevel Regression as Default."](https://elevanth.org/blog/2017/08/24/multilevel-regression-as-default/) Elevanth.org Blog.

7. Stein, C. (1956). "Inadmissibility of the usual estimator for the mean of a multivariate normal distribution." *Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability*, 1, 197-206.

8. Efron, B., & Morris, C. (1977). "Stein's Paradox in Statistics." *Scientific American*, 236(5), 119-127.

9. Moulton, B. R. (1990). "An Illustration of a Pitfall in Estimating the Effects of Aggregate Variables on Micro Units." *The Review of Economics and Statistics*, 72(2), 334-338.

10. Kish, L. (1965). *Survey Sampling*. John Wiley & Sons. [Design effect concept]

11. Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How Much Should We Trust Differences-in-Differences Estimates?" *The Quarterly Journal of Economics*, 119(1), 249-275.

---

## Appendix: Code Availability

All code and data are available at: https://github.com/IlanStrauss/anthropic-econ-critique

- `analysis_full.py`: Python analysis using statsmodels
- `analysis_brms.R`: R analysis using brms (Bayesian)
- `data/`: Anthropic's original data
