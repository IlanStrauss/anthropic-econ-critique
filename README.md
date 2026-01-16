# Critique of Anthropic Economic Index

**Authors**: Ilan Strauss
**AI Disclosures Project**

---

## 📖 Read Our Analysis

| Format | Link | Description |
|--------|------|-------------|
| **🌐 Blog (Easy Read)** | [**View Online**](https://ilanstrauss.github.io/anthropic-econ-critique/) | Visual, accessible version with figures |
| **📝 Background Note** | [PARTIAL_POOLING.md](PARTIAL_POOLING.md) | Detailed methodology, equations, citations |
| **✍️ Blog Post (Markdown)** | [BLOG_POST.md](BLOG_POST.md) | Markdown version of the blog |

---

## Overview

Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) claims "Worldwide, uneven adoption remains well-explained by GDP per capita."

**Using their own data, we show this is not true for middle-income countries—where much of the world's population resides.**

Using Anthropic's own data with partial pooling models, we show:
1. **The income-AI adoption relationship varies dramatically by income level** — from 0.44 (middle-income) to 0.76 (low-income)
2. **Middle-income countries adopt AI beyond what their income predicts** — education, infrastructure, and policy drive adoption
3. **Their uncertainty is underestimated** (~3x) due to ignoring country-level heterogeneity
4. **Divergence is not inevitable** — their cross-sectional estimate does not imply dynamics

## Key Findings

See [PARTIAL_POOLING.md](PARTIAL_POOLING.md) for full write-up.

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

Anthropic's analysis implies a simple story: **higher income → more AI adoption**. Our findings show this varies dramatically.

### Their Story vs. Our Story

| Anthropic's Claim | Our Finding | Implication |
|-------------------|-------------|-------------|
| "Worldwide, uneven adoption remains well-explained by GDP per capita" | Not true for middle-income countries (R²=0.14) | Most of the world's population does not fit their story |
| Single global elasticity of 0.70 | Elasticity varies from 0.44 to 0.76 | One-size-fits-all policy is inappropriate |
| Tight confidence interval [0.61, 0.77] | True interval is [0.43, 0.89] | We're far less certain about this relationship than they suggest |

### A Striking Example: South Korea vs USA

The United States has a GDP per capita of $132,532—**2.6 times** South Korea's $51,496. Yet South Korea's AI Usage Index (3.73) is actually *slightly higher* than the USA's (3.62). If GDP per capita were the primary driver of AI adoption, the USA should have dramatically higher adoption. It does not. Education, digital infrastructure, and cultural factors clearly matter more than income alone.

### What Should Policymakers Know?

**For low-income countries (β = 0.76):**
- Income level IS strongly associated with AI adoption
- Economic development may be a necessary precondition
- Focus: basic infrastructure, connectivity, economic fundamentals

**For middle-income countries (β = 0.44):**
- **These countries adopt AI beyond what their income predicts** — this is the critical finding
- Education, English proficiency, digital literacy, tech infrastructure, and regulatory environment are driving adoption now
- Examples: Brazil, Mexico, Thailand, Malaysia don't need to wait to get richer

**For high-income countries (β = 0.63):**
- Massive variation unexplained by income
- Israel is a 3x over-adopter; Gulf states are under-adopters despite wealth
- Cultural, linguistic, and policy factors likely dominate

### The Middle-Income Story

Our most striking finding: middle-income countries are adopting AI beyond what their wealth would predict. Unlike low-income countries (where income level strongly predicts adoption) or high-income countries (where adoption is already high), middle-income nations are finding other pathways to AI adoption.

This is good news: these countries don't need to wait to get richer. Education, infrastructure, and policy can drive adoption now.

### Divergence Is Not Inevitable

Anthropic warns of "divergence in living standards." But a cross-sectional income-AI elasticity does not imply divergence. Divergence requires a dynamic feedback loop: GDP growth → AI adoption growth → productivity growth → more GDP growth. Anthropic only estimates the first link (income levels → AI adoption levels). The divergence claim is speculation beyond their data.

## Repository Structure

```
anthropic-econ-critique/
├── README.md                    # This file
├── PARTIAL_POOLING.md                  # Full write-up
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
