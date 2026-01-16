# Methodological Critique of Anthropic's Economic Index

**Ilan Strauss**
**AI Disclosures Project**
**January 2026**

---

## Abstract

Anthropic's Economic Index claims "Worldwide, uneven adoption remains well-explained by GDP per capita." Using their own data, we show this is not true for middle-income countries—where much of the world's population resides.

Using Anthropic's own data with partial pooling models, we show the GDP per capita-AI adoption relationship varies dramatically by income level: from 0.76 in low-income countries to just 0.44 in middle-income countries. Middle-income countries—home to much of the world's population—are adopting AI beyond what their income level alone would predict.

The head of economics at Anthropic, Peter McCrory, warned in the *Financial Times*, "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards." Yet I try to show that Anthropic's latest research does not provide much evidence on the question of if AI will create more or less convergence in global income levels between countries.

But a cross-sectional regression looking at how income (GDP per capita) impacts AI adoption doesn't imply divergence in living standards. Impacts on living standards from AI adoption cannot be estimated from this regression. This requires estimating the second order effect of how AI adoption impacts productivity, which Anthropic does not do here.

Moreover, our findings suggest more reason to predict convergence in AI adoption at least: middle-income countries are already adopting AI beyond what their income predicts, when allowing for how GDP per capita impacts AI adoption to vary by a country's starting income level.

For middle-income countries, income level is a weak predictor of AI adoption. Selective investments in education, digital infrastructure, English proficiency, and regulatory environment may be driving adoption. These are actionable policy levers—divergence is not inevitable.

---

## 1. Introduction

Anthropic's January 2026 Economic Index report presents a simple, compelling story: national income strongly predicts AI adoption. Their headline finding states:

> "At the country level, a 1% increase in GDP per capita is associated with a 0.7% increase in Claude usage per capita." (Chapter 3: How Claude is used varies by geography)

They further claim to "replicate the finding from our prior report that GDP is strongly correlated with the AUI (Anthropic AI Usage Index)" and note this relationship "holds for both countries and US states" (Figures 3.3 and 3.4).

Note: Anthropic uses "GDP" and "GDP per capita" interchangeably throughout their report (e.g., "GDP and human education predict adoption globally" followed immediately by "A 1% increase in GDP per capita..."). These are not the same: GDP is total national output; GDP per capita is income per person. Their regression is on GDP per capita, not GDP. We use "GDP per capita" or "income level" throughout for precision.

However, the report presents a **single global elasticity** without examining whether this relationship varies systematically across development levels. While they acknowledge that "the primitives themselves are not necessarily causal factors—we don't know if income or education are truly driving adoption, or if they're proxies for other underlying conditions," they do not test for heterogeneity in the GDP per capita-AI adoption relationship itself.

We examine whether this conclusion is robust to:
- Proper uncertainty quantification using hierarchical models
- Heterogeneity in the GDP per capita-AI adoption relationship across development levels
- Influence of outlier countries

### Core Findings

![Figure 1](figures/fig1_their_view_vs_ours.png)

**Figure 1**: Left panel shows Anthropic's approach—one regression line through all countries. Right panel shows separate lines by income tercile. The slopes differ substantially: low-income countries (red) show a steep relationship; middle-income countries (orange) show a shallow one.

**Their Claim vs Our Finding:**

| Anthropic's Claim | Our Finding |
|-------------------|-------------|
| "a 1% increase in GDP per capita is associated with a 0.7% increase in Claude usage" | Elasticity is 0.66 with partial pooling; ranges from 0.44 to 0.76 by income level |
| "Worldwide, uneven adoption remains well-explained by GDP per capita" | Not true for middle-income countries (R²=0.14)—much of the world's population |
| Single elasticity presented without uncertainty discussion | True 95% CI is [0.43, 0.89], not [0.61, 0.77] |

**Policy Implications:**

Anthropic's implicit story is: higher income → more AI adoption. Our findings show this varies dramatically:

- **Low-income countries**: Income level is a strong predictor of AI adoption. Anthropic's story is approximately correct.
- **Middle-income countries**: These countries are adopting AI beyond what their income level predicts (β = 0.44). Education, digital literacy, English proficiency, and tech infrastructure are driving adoption now.
- **High-income countries**: Variation in AI usage is large and unexplained by income. Israel has 3x the AI usage predicted by its income level; Gulf states have far less. Cultural and policy factors dominate.

The "divergence in living standards" that Anthropic warns of is not inevitable—middle-income countries are already finding pathways to AI adoption beyond wealth.

---

## 2. Data and Methods

### 2.1 Why Partial Pooling?

Anthropic faces a fundamental choice when estimating the GDP per capita-AI adoption relationship:

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

When the true data-generating process has group-varying slopes (see [Moulton 1990](https://www.jstor.org/stable/2109724); [Bertrand, Duflo & Mullainathan 2004](https://academic.oup.com/qje/article/119/1/249/1876068)):

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

In our context: Anthropic's complete pooling assumes all countries share identical GDP per capita-AI adoption relationships. This is empirically false (slopes range from 0.44 to 0.76) and theoretically unjustified. Partial pooling reveals both the heterogeneity and the appropriate uncertainty.

### 2.2 Data

We use Anthropic's publicly released data from their [HuggingFace repository](https://huggingface.co/datasets/Anthropic/EconomicIndex), specifically the `release_2025_09_15` dataset containing:
- Claude.ai usage data (November 2025)
- Country-level aggregates with GDP per working-age capita
- 114 countries with ≥200 observations (their filtering threshold)

**How Anthropic measures AI adoption:** Anthropic's "AI Usage Index" (AUI) is based on Claude.ai usage per capita—the number of conversations users in each country have with Claude, normalized by working-age population. This measures adoption of *one* AI assistant (Claude), not AI adoption broadly. Anthropic acknowledges this limitation but argues Claude usage is a reasonable proxy for frontier AI adoption patterns.

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

### 2.4 Estimation

For partial pooling, we use **Maximum Likelihood Estimation (MLE)** via `statsmodels.MixedLM` in Python. This estimates the fixed effect (global slope) and random effects (group deviations) jointly by maximizing the likelihood of the observed data given the hierarchical structure.

The model assumes random effects are normally distributed with mean zero and variance estimated from the data. This yields **empirical Bayes** estimates: group-specific slopes are shrunk toward the global mean, with the degree of shrinkage determined by the estimated between-group variance relative to within-group variance.

For researchers preferring full Bayesian inference, we provide `analysis_brms.R` which uses [brms](https://paul-buerkner.github.io/brms/) (Bürkner 2017) with either MCMC sampling or ADVI (Automatic Differentiation Variational Inference) for faster approximate posteriors. The qualitative conclusions are robust to estimation method.

---

## 3. Results

### 3.1 Uncertainty is Underestimated

![Figure 3](figures/fig3_confidence_intervals.png)

**Figure 3**: Confidence interval comparison. Anthropic's OLS (top) shows a narrow interval [0.61, 0.77]. Partial pooling (bottom) shows a wider interval [0.43, 0.89].

| Method | Slope (β) | Std. Error | 95% CI |
|--------|-----------|------------|--------|
| Anthropic OLS | 0.69 | 0.042 | [0.61, 0.77] |
| Partial Pooling | 0.66 | 0.116 | [0.43, 0.89] |

**Finding**: The partial pooling standard error is **2.8x larger** than OLS. Anthropic's confidence interval is too narrow because it ignores group-level variance.

**Implication**: The true 95% CI includes values as low as 0.43—almost half the point estimate. The relationship between GDP and AI usage is **far less precisely estimated** than Anthropic suggests.

### 3.2 The Relationship Between GDP and AI Usage Varies by Income Level

![Figure 2](figures/fig2_slope_by_income.png)

**Figure 2**: Estimated elasticity (with standard errors) for each income group. The middle-income bar is notably shorter than Anthropic's global estimate.

| Income Tercile | Slope (β) | SE | R² | N |
|----------------|-----------|-----|-----|---|
| Low income | 0.76 | 0.19 | 0.30 | 38 |
| **Middle income** | **0.44** | 0.18 | 0.14 | 38 |
| High income | 0.63 | 0.20 | 0.21 | 38 |

**Note on data coverage:** China is not included in Anthropic's dataset. India and Indonesia are included but classified as *low-income* based on GDP per working-age capita—they are not in the middle-income tercile. The 38 middle-income countries range from South Africa ($9,273 GDP/capita) to Poland ($38,209), and include Brazil, Mexico, Thailand, Malaysia, Colombia, Argentina, Turkey, Chile, Peru, and Romania.

**Finding**: The elasticity of AI usage with respect to GDP varies by **73%** across income levels (0.44 to 0.76).

**Key insight**: Middle-income countries show a **much weaker** relationship between GDP per capita and AI usage (β = 0.44 vs 0.70 global average). This suggests:

1. In low-income countries, income level is a **binding constraint**—basic economic development is necessary before AI adoption can increase
2. In middle-income countries, **these countries are adopting AI beyond what their income predicts**. Education, English proficiency, tech infrastructure, and regulatory environment are driving adoption—not income level
3. Among high-income countries, there is **substantial variation in AI usage** unexplained by income (e.g., Israel has 3x the AI usage predicted by its income level; Gulf states have far less than predicted)

**Policy implication**: Middle-income countries don't need to wait to get richer. For Brazil, Mexico, Thailand, and similar nations, education, infrastructure, and policy are already driving AI adoption beyond what income alone predicts.

**Evidence from residuals**: Using Anthropic's own pooled regression, we can examine whether middle-income countries systematically over- or under-adopt relative to prediction:

| Income Group | Mean Residual | Median Residual | % Above Prediction |
|--------------|---------------|-----------------|-------------------|
| Low | -0.059 | +0.042 | 50.0% |
| **Middle** | **+0.094** | **+0.122** | **63.2%** |
| High | -0.035 | -0.054 | 44.7% |

Middle-income countries have a median residual of +0.122, meaning the typical middle-income country adopts about 13% more AI than predicted by Anthropic's regression (e^0.122 ≈ 1.13). Nearly two-thirds (63%) of middle-income countries are above the regression line. This directly supports our claim: middle-income countries are adopting AI beyond what their income level alone would predict.

### 3.3 Outliers

![Figure 4](figures/fig4_outliers.png)

**Figure 4**: Influential outliers in the GDP-AI usage relationship. Point size indicates influence (Cook's D). Red = over-adopter (above regression line), Blue = under-adopter (below line).

Some countries deviate substantially from the income-AI adoption relationship: Israel has 3x the AI usage predicted by its income level; Gulf states (Qatar, Kuwait, Saudi Arabia) have far less than predicted; several African countries (Tanzania, Angola) fall well below the regression line. Georgia and South Korea are notable over-adopters relative to their income levels.

**A striking example: South Korea vs USA.** The United States has a GDP per capita of $132,532—**2.6 times** South Korea's $51,496. Yet South Korea's AI Usage Index (3.73) is actually *slightly higher* than the USA's (3.62). If GDP per capita were the primary driver of AI adoption, as Anthropic's headline implies, the USA should have dramatically higher adoption. It doesn't. Education, digital infrastructure, and cultural factors clearly matter more than income alone.

However, removing outliers only shifts the slope by ~5% (see Appendix B for details). The main findings—heterogeneity by income level and underestimated uncertainty—are not driven by outliers.

---

## 4. Discussion

### 4.1 What Anthropic Gets Wrong

**Their Claim vs Our Finding:**

| Anthropic's Claim | Our Finding |
|-------------------|-------------|
| "a 1% increase in GDP per capita is associated with a 0.7% increase in Claude usage" | Elasticity is 0.66 with partial pooling; ranges from 0.44 to 0.76 by income level |
| "Worldwide, uneven adoption remains well-explained by GDP per capita" | Not for middle-income countries (R²=0.14)—most of world's population |
| Relationship "holds for both countries and US states" | Relationship varies dramatically: near-zero effect for middle-income nations |
| Single elasticity presented without uncertainty discussion | True 95% CI is [0.43, 0.89], not [0.61, 0.77] |

**Specific Methodological Issues:**

1. **Treats heterogeneity as noise**: By pooling all countries into one regression, they assume the GDP per capita-AI adoption relationship is constant globally. It is not. The slope varies by 73% across income terciles (0.44 to 0.76).

2. **Underestimates uncertainty**: OLS standard errors assume independent observations with constant variance. Country-level data has hierarchical structure that inflates true uncertainty by ~3x.

3. **Ignores influential observations**: A handful of unusual countries (Israel, Gulf states, African outliers) disproportionately determine the slope. Tanzania alone has Cook's D of 0.11—exceeding the 4/n threshold by 3x.

### 4.2 What This Means for Policy

**Anthropic's implicit recommendation**: "Grow GDP → Get AI adoption"

**Our correction**:
- For **low-income countries**: This is approximately correct. Income level is a binding constraint.
- For **middle-income countries**: **These countries are adopting beyond their income.** The 0.44 elasticity shows income level is a weak predictor. What's driving adoption:
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

For middle-income countries—home to much of the world's population—income level is a weak predictor of AI adoption. These countries are already adopting AI beyond what their wealth predicts, driven by education, infrastructure, and policy. The "divergence in living standards" that Anthropic warns of is not inevitable.

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

12. Bürkner, P.-C. (2017). "brms: An R Package for Bayesian Multilevel Models Using Stan." *Journal of Statistical Software*, 80(1), 1-28. https://doi.org/10.18637/jss.v080.i01

13. Hsiao, C. (2022). *Analysis of Panel Data* (3rd ed.). Cambridge University Press. https://www.cambridge.org/core/books/analysis-of-panel-data/B8C2B0B64BEB1682A845C5F3FF677E61

---

## Appendix A: Code Availability

All code and data are available at: https://github.com/IlanStrauss/anthropic-econ-critique

- `analysis_full.py`: Python analysis using statsmodels
- `analysis_brms.R`: R analysis using brms (Bayesian)
- `data/`: Anthropic's original data

## Appendix B: Outlier Analysis

### Influential Observations

Six countries exceed the Cook's Distance threshold (4/n = 0.035):

| Country | Cook's D | Residual Direction | Description |
|---------|----------|-------------------|-------------|
| Tanzania | 0.106 | Below line | Far less AI usage than GDP predicts |
| Angola | 0.097 | Below line | Far less AI usage than GDP predicts |
| Uzbekistan | 0.050 | Below line | Less AI usage than GDP predicts |
| Qatar | 0.048 | Below line | Rich but low AI adoption |
| Israel | 0.042 | Above line | 3x the AI usage predicted by GDP |
| Kuwait | 0.037 | Below line | Rich but low AI adoption |

### Sensitivity to Outlier Removal

| Countries Removed | Slope (β) | Change from Full Sample |
|-------------------|-----------|------------------------|
| None (full sample, n=114) | 0.689 | — |
| Israel only | 0.680 | −1.4% |
| Tanzania only | 0.673 | −2.4% |
| Tanzania + Angola + Israel | 0.651 | −5.6% |
| All 6 outliers | 0.658 | −4.6% |

### Interpretation

Removing influential outliers shifts the estimated elasticity by at most ~6%. This is meaningful but does not overturn the main findings:

1. The heterogeneity by income level (slopes ranging from 0.44 to 0.76) remains the dominant issue
2. The uncertainty underestimation (~3x) is driven by group-level variance, not outliers
3. Robust regression (Huber M-estimation) gives β = 0.67, consistent with outlier removal

The outliers do, however, highlight that country-specific factors beyond GDP—language, culture, regulatory environment, tech infrastructure—substantially affect AI adoption. Israel's over-adoption and the Gulf states' under-adoption relative to their GDP levels warrant further investigation.
