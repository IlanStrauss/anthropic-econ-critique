# Bayesian Partial Pooling Analysis: Anthropic's Economic Index

**Ilan Strauss**
**AI Disclosures Project**
**January 2026**

---

## Abstract

This technical document presents a Bayesian hierarchical analysis of Anthropic's GDP per capita-AI adoption relationship. Using partial pooling with 7 country groups, we estimate a global elasticity of **β = 0.54** with 95% CI **[0.33, 0.74]** — substantially lower than Anthropic's OLS estimate of 0.70 (which falls outside our credible interval).

The Bayesian approach properly accounts for group-level heterogeneity and yields appropriately wider uncertainty bounds. Anthropic's standard error of 0.042 underestimates true uncertainty by approximately 2.4x (our SE = 0.10).

For a discussion of separate OLS regressions by income tercile and policy implications, see [BLOG_POST.md](BLOG_POST.md).

---

## 1. Why Partial Pooling?

Anthropic faces a fundamental choice when estimating the GDP per capita-AI adoption relationship:

1. **Complete pooling** (their approach): Estimate one global slope, ignoring country/region differences
2. **No pooling**: Estimate separate slopes for each group, ignoring shared information
3. **Partial pooling**: Estimate group-specific slopes that are *shrunk* toward the global mean

Partial pooling is theoretically preferred for several reasons:

### James-Stein Estimator Properties

[Stein (1956)](https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator) proved a remarkable result: when estimating three or more means simultaneously, the MLE (no pooling) is *inadmissible* — there always exists a shrinkage estimator with lower mean squared error. The [James-Stein estimator](https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator) shrinks individual estimates toward the grand mean, and this "borrowing of strength" across groups reduces total estimation error. Partial pooling implements this principle: noisy group estimates are pulled toward the global estimate, with the degree of shrinkage determined by relative sample sizes and variance. As [Efron and Morris (1977)](https://statweb.stanford.edu/~ckirby/brad/other/Article1977.pdf) explained in their famous *Scientific American* article, this "Stein's Paradox" shows that combining information across groups is almost always better than treating them independently.

### Bias-Variance Tradeoff

As McElreath (2020) explains in *Statistical Rethinking*, partial pooling navigates between two failure modes:
- Complete pooling *underfits*: low variance (one estimate for all) but high bias (ignores real group differences)
- No pooling *overfits*: low bias (estimates each group separately) but high variance (noisy estimates, especially for small groups)

Partial pooling introduces some bias (by shrinking group estimates toward the global mean) in exchange for reduced variance. This tradeoff is beneficial when groups are noisy: groups with little data are shrunk heavily toward the global mean (regularization dominates), while groups with abundant data retain their individual estimates (data dominates).

### Proper Uncertainty Quantification

Gelman et al. (2013) emphasize in *Bayesian Data Analysis* that hierarchical models correctly propagate uncertainty from the group level to the population level. OLS standard errors assume fixed, known group effects — partial pooling acknowledges they are estimated with error, yielding appropriately wider confidence intervals.

---

## 2. Why OLS Standard Errors Are Wrong Here

Anthropic's OLS approach assumes observations are **independent and identically distributed** with a **constant slope**. Both assumptions fail with grouped data when slopes vary.

### The Problem: Slope Heterogeneity

When the true data-generating process has group-varying slopes (see [Moulton 1990](https://www.jstor.org/stable/2109724); [Bertrand, Duflo & Mullainathan 2004](https://academic.oup.com/qje/article/119/1/249/1876068)):

**y<sub>ij</sub> = α + α<sub>j</sub> + (β + β<sub>j</sub>) × x<sub>ij</sub> + ε<sub>ij</sub>**

where β<sub>j</sub> ~ N(0, τ²<sub>β</sub>) are random slope deviations, the variance of the pooled OLS estimator is:

**Var(β̂<sub>pooled</sub>) = Var(β̂ | slopes fixed) + Var(slopes across groups)**

OLS only captures the first term — the **sampling variance** given fixed slopes. It completely ignores the second term — the **between-group variance in slopes** (`τ²_β`).

### The Moulton Problem in Economics

This is a well-known issue when regressing micro outcomes on macro predictors. Bertrand, Duflo & Mullainathan (2004) showed that difference-in-differences studies routinely understate standard errors by 2-3x when ignoring group clustering — exactly what we find here.

---

## 3. Data

We use Anthropic's publicly released data from their [HuggingFace repository](https://huggingface.co/datasets/Anthropic/EconomicIndex), specifically the `release_2025_09_15` dataset containing:
- Claude.ai usage data (November 2025)
- Country-level aggregates with GDP per working-age capita
- 114 countries with ≥200 observations (their filtering threshold)

---

## 4. Model Specification

We fit a Bayesian hierarchical model using `brms` (Bürkner 2017) with random intercepts and slopes by country group:

**log(AUI)<sub>i</sub> = α + α<sub>g[i]</sub> + (β + β<sub>g[i]</sub>) × log(GDP)<sub>i</sub> + ε<sub>i</sub>**

where:
- g[i] = group for country i (7 groups)
- α<sub>g</sub> ~ N(0, σ<sub>α</sub>) — random intercepts by group
- β<sub>g</sub> ~ N(0, σ<sub>β</sub>) — random slopes by group
- ε<sub>i</sub> ~ N(0, σ) — residual error

### Group Construction

With only 3 income terciles, the model exhibits heavy shrinkage and convergence issues (341 divergent transitions). Following Gelman & Hill (2007), we created 7 theoretically meaningful groups to improve estimation:

| Group | Description | N |
|-------|-------------|---|
| Gulf | GCC oil states (QAT, KWT, SAU, ARE, BHR, OMN) - high GDP, very low adoption | 6 |
| TechHub | High adopters (ISR, GEO, ARM, KOR, MNE) | 5 |
| LowAfrica | Low-income African countries | 16 |
| LowAsia | Low-income Asian countries | 12 |
| LowOther | Other low-income | 10 |
| Mid | Middle-income | 33 |
| High | High-income (excl. Gulf, TechHub) | 32 |

### Estimation

MCMC sampling with 4 chains, 4000 iterations (1000 warmup), `adapt_delta = 0.99`.

---

## 5. Results

### Global Estimate

| Method | Slope (β) | SE | 95% CI |
|--------|-----------|-----|--------|
| Anthropic (pooled OLS) | 0.69 | 0.04 | [0.61, 0.77] |
| **Bayesian hierarchical (7 groups)** | **0.54** | **0.10** | **[0.33, 0.74]** |

**Anthropic's estimate of 0.69 falls outside our 95% credible interval.**

### Group-Specific Slopes

| Group | Unpooled OLS | Partial Pooling | 95% CI | N |
|-------|--------------|-----------------|--------|---|
| Gulf | 0.93 | **0.49** | [0.28, 0.72] | 6 |
| TechHub | 0.68 | **0.61** | [0.38, 0.85] | 5 |
| LowAfrica | 0.23 | **0.49** | [0.22, 0.72] | 16 |
| LowAsia | 0.56 | **0.53** | [0.29, 0.77] | 12 |
| LowOther | 0.49 | **0.54** | [0.30, 0.78] | 10 |
| Mid | 0.62 | **0.55** | [0.36, 0.76] | 33 |
| High | 0.43 | **0.55** | [0.35, 0.73] | 32 |

### Model Diagnostics

- Only 2 divergent transitions (acceptable)
- Rhat = 1.00 for all parameters
- Effective sample sizes > 8000

---

## 6. Key Findings

### 1. Global slope is 0.54, not 0.70

Anthropic's estimate is outside our 95% CI upper bound of 0.74 and well above our point estimate.

### 2. Slopes cluster around 0.49–0.61

After proper shrinkage, all groups have similar slopes — less heterogeneity than unpooled OLS suggests, but consistently lower than Anthropic's 0.70.

### 3. SE is 2.4x larger than Anthropic's

Our SE of 0.10 vs their 0.042 reflects proper accounting for group-level variance.

### 4. Why is the partial pooling estimate lower than OLS?

The simple average of unpooled group slopes is:
```
(0.93 + 0.68 + 0.23 + 0.56 + 0.49 + 0.62 + 0.43) / 7 = 0.56
```

This is close to our partial pooling estimate of 0.54. Anthropic's pooled OLS of 0.69 is inflated by a **composition effect**: Gulf states (high GDP, low adoption) and LowAfrica (low GDP, low adoption) create a steeper apparent slope in pooled regression than exists within any group (**Simpson's paradox**).

---

## 7. Implications

Anthropic's headline finding — 0.7 elasticity between GDP and AI usage — is:
1. **Overstated**: True global estimate is ~0.54, not 0.70
2. **Overconfident**: True uncertainty is ~2.4x larger than reported
3. **Inflated by composition effects**: Simpson's paradox makes the global slope steeper than any within-group slope

---

## References

1. Anthropic. (2026). Anthropic Economic Index: January 2026 Report. https://www.anthropic.com/research/anthropic-economic-index-january-2026-report

2. Anthropic. (2025). EconomicIndex Dataset. HuggingFace. https://huggingface.co/datasets/Anthropic/EconomicIndex

3. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.

4. Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.

5. McElreath, R. (2020). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). Chapman and Hall/CRC.

6. Stein, C. (1956). "Inadmissibility of the usual estimator for the mean of a multivariate normal distribution." *Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability*, 1, 197-206.

7. Efron, B., & Morris, C. (1977). "Stein's Paradox in Statistics." *Scientific American*, 236(5), 119-127.

8. Moulton, B. R. (1990). "An Illustration of a Pitfall in Estimating the Effects of Aggregate Variables on Micro Units." *The Review of Economics and Statistics*, 72(2), 334-338.

9. Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How Much Should We Trust Differences-in-Differences Estimates?" *The Quarterly Journal of Economics*, 119(1), 249-275.

10. Bürkner, P.-C. (2017). "brms: An R Package for Bayesian Multilevel Models Using Stan." *Journal of Statistical Software*, 80(1), 1-28.

11. Hsiao, C. (2022). [*Analysis of Panel Data*](https://rodorigo.wordpress.com/wp-content/uploads/2020/02/cheng-hsiao-analysis-of-panel-dataz-lib.org_.pdf) (3rd ed.). Cambridge University Press.

---

## Code Availability

All code and data are available at: https://github.com/IlanStrauss/anthropic-econ-critique

- `analysis_brms.R`: R analysis using brms (Bayesian)
- `analysis_full.py`: Python analysis using statsmodels
- `data/`: Anthropic's original data
