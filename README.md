# Critique of Anthropic Economic Index

**Authors**: Ilan Strauss & Tim O'Reilly
**AI Disclosures Project**

## Overview

This repository contains a methodological critique of Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report), focusing on their claim that "a 1% increase in GDP per capita is associated with a 0.7% increase in Claude usage per capita."

We show that:
1. **Their uncertainty is underestimated** (~3x) due to ignoring country-level heterogeneity
2. **The GDP-usage relationship varies dramatically by income level** - from 0.44 (middle-income) to 0.76 (low-income)
3. **A handful of outliers drive their results** (Israel, Gulf states, African countries)

## Key Findings

See [CRITIQUE.md](CRITIQUE.md) for full write-up.

### Effect Size Comparison

| Method | GDP Elasticity (β) | Std. Error | 95% CI |
|--------|-------------------|------------|--------|
| Anthropic OLS | 0.70 | 0.042 | [0.61, 0.77] |
| Partial Pooling | 0.66 | 0.116 | [0.43, 0.89] |
| Robust (Huber) | 0.67 | 0.040 | [0.59, 0.75] |

### Slope Heterogeneity by Income Level

| Income Tercile | Slope (β) | Std. Error | N |
|----------------|-----------|------------|---|
| Low income | 0.76 | 0.19 | 38 |
| **Mid income** | **0.44** | 0.18 | 38 |
| High income | 0.63 | 0.20 | 38 |

## Why This Matters: Policy Implications

Anthropic's analysis implies a simple policy story: **grow GDP → get AI adoption**. Our findings reveal this is dangerously oversimplified.

### Their Story vs. Our Story

| Anthropic's Claim | Our Finding | Implication |
|-------------------|-------------|-------------|
| "GDP strongly predicts AI usage" | Relationship is weak (R²=0.14) in middle-income countries | GDP growth alone won't close the AI gap for most of the world's population |
| Single global elasticity of 0.70 | Elasticity varies from 0.44 to 0.76 | One-size-fits-all policy is inappropriate |
| Tight confidence interval [0.61, 0.77] | True interval is [0.43, 0.89] | We're far less certain about this relationship than they suggest |

### What Should Policymakers Actually Do?

**For low-income countries (β = 0.76):**
- GDP growth IS strongly associated with AI adoption
- Economic development may be a necessary precondition
- Focus: basic infrastructure, connectivity, economic fundamentals

**For middle-income countries (β = 0.44):**
- **GDP barely matters** — this is the critical finding
- Other factors dominate: education, English proficiency, digital literacy, tech infrastructure, regulatory environment
- Focus: human capital investment, not just GDP growth
- Examples: Brazil, Mexico, Thailand, Indonesia — telling these countries to "grow GDP for AI" misses the point

**For high-income countries (β = 0.63):**
- Massive variation unexplained by GDP
- Israel is a 7x over-adopter; Gulf states are under-adopters despite wealth
- Cultural, linguistic, and policy factors likely dominate
- Focus: understand what makes Israel succeed vs. why wealth alone doesn't predict adoption

### The Middle-Income AI Trap

Our most striking finding is that middle-income countries face a kind of **AI adoption trap**. Unlike low-income countries (where GDP growth helps) or high-income countries (where adoption is already high), middle-income nations can't simply grow their way to AI adoption.

This has echoes of the broader "middle-income trap" in development economics — and suggests AI policy for these countries needs to be fundamentally different from the "GDP drives everything" narrative.

### Uncertainty Matters for Policy

Anthropic's narrow confidence interval [0.61, 0.77] excludes 0.44 as implausible. But 0.44 is exactly the elasticity we observe in middle-income countries. Their overconfidence could lead policymakers to dismiss evidence that contradicts the "GDP → AI" story — when that evidence is exactly what should inform middle-income country strategy.

## Repository Structure

```
anthropic-econ-critique/
├── README.md                    # This file
├── CRITIQUE.md                  # Full write-up
├── analysis_full.py             # Main Python analysis
├── analysis_brms.R              # R/brms Bayesian analysis
├── analysis_results.csv         # Processed results
├── data/                        # Anthropic's original data
│   └── release_2025_09_15/
├── original_report/             # Their report
└── figures/                     # Generated figures
```

## Reproducing Results

### Python
```bash
pip install pandas numpy statsmodels scipy
python analysis_full.py
```

### R (brms)
```r
install.packages(c("brms", "tidyverse"))
source("analysis_brms.R")
```

## Data Source

Original data from Anthropic's HuggingFace repository:
- https://huggingface.co/datasets/Anthropic/EconomicIndex

## Contact

- Ilan Strauss: ilan@aidisclosures.org
- AI Disclosures Project: https://aidisclosures.org
