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
