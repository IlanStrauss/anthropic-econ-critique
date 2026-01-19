# Does GDP per Capita Really Predict AI Adoption? Adjusting Anthropic's Estimates
## A Critique

**Authors**: Ilan Strauss
**AI Disclosures Project** (v.2)

---

## Summary: Anthropic's Claims vs. Our Critique

| Anthropic's Claim | Critique | Our Evidence |
|-------------------|--------------|--------------|
| ["Uneven [AI] adoption remains well-explained by GDP per capita"](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) (global β=0.71) | GDP per capita explains little of middle-income country adoption | Middle-income group: R²=7% (with Seychelles), R²=17% (without Seychelles) |
| AI usage concentration ["essentially unchanged"](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) (Aug–Nov 2025) | Gini captures a static snapshot; differential growth rates compound into widening gaps | Adoption changes across two time periods show potential inequality emerging: High-income +26%, Middle-income +14%, Low-income +22% |
| ["Divergence in living standards" possible](https://www.ft.com/content/3ad44e30-c738-4356-91fb-8bb2368685c4) (McCrory to FT) | Speculative: no second-order effects estimated (AI → productivity → growth) | Cross-country data used is a snapshot from Claude front-endusage  (closer to consumer), not API (closer to firm-level), and not panel (countries over time)|
| Implied by FT coverage: data reflects global AI adoption | Data only measures Claude usage, not total AI adoption broadly | Brazil and Thailand saw *decreased* Claude usage. But could be from competitive use of alternatives (ChatGPT), not reduced AI adoption |
| Single global relationship applies to all countries | Different income groups have different relationships, making pooling biased (Hsiao 2022) | Separate regressions show relationships (slopes) and R² (fit) vary by income group |

---

## 📖 Read Our Analysis

| Format | Link | Description |
|--------|------|-------------|
| **✍️ Blog Post** | [**BLOG_POST.md**](BLOG_POST.md) | Full blog post |
| **🌐 Blog (Easy Read)** | [View Online](https://ilanstrauss.github.io/anthropic-econ-critique/) | Simplified analysis |

---

## Overview

We take issue with two claims or inferences:

1) *CORE ANTHROPIC CLAIM 1*: Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) claims **"Worldwide, uneven [AI] adoption remains well-explained by GDP per capita."** 

- We show that the data does not support this. When broken down by income-group the fit is substantailly weaker and non-existant for middle-income countries, roughly one-third of the country sample.

2) *CORE FINANCIAL TIMES CLAIM 2*: The [*Financial Times*](https://www.ft.com/content/3ad44e30-c738-4356-91fb-8bb2368685c4) covered Anthropic's research with the headline: **"Rich countries’ greater use of AI risks deepening inequality, Anthropic warns".**

- This is misleading since none of the Anthropic research covered in the article in question provides evidence on this claim. The head of economics at Anthropic [warns in the article](https://www.ft.com/content/3ad44e30-c738-4356-91fb-8bb2368685c4): "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards." 

- It is unclear how this relates to the research they have conducted which analyzes adoption of Claude, focusing on consumer adoption not firm-level (API), and not attempting to assess if changing adoption patterns reflects greater usage of ChatGPT or competing products instead


### Weak and Highly Uncertain Relationship with Little Explanatory Power

Using two data releases from Anthropic (August 4-11 and November 13-20, 2025), *we show that the GDP per capita - AI adoption relationship is weak and highly uncertain*, across both time periods:

**Table: Regression Results — Anthropic's Global Estimate vs. Our Country-Group Estimates**

| Income Group | Period | Slope (β) | Std. Error | p-value | Significant? | R² | N |
|--------------|--------|-----------|------------|---------|--------------|-----|---|
| **Global** | Aug 4-11 | 0.69 | 0.04 | <0.001 | Yes | 0.71 | 114 |
| | Nov 13-20 | 0.71 | 0.06 | <0.001 | Yes | 0.56 | 116 |
| **Low income** | Aug 4-11 | 0.76 | 0.19 | <0.001 | Yes | 0.30 | 38 |
| | Nov 13-20 | 0.85 | 0.18 | <0.001 | Yes | 0.37 | 39 |
| **Middle income** | Aug 4-11 | 🔴 0.44 | 0.18 | 0.019 | Yes | 0.14 | 38 |
| | **Nov 13-20 (with Seychelles)** | **0.73** | **0.44** | **0.105** | **No (~10%)** | **0.07** | 38 |
| | Nov 13-20 (excl. Seychelles) | 🔴 0.44 | 0.16 | 0.011 | Yes | 0.17 | 37 |
| **High income** | Aug 4-11 | 0.63 | 0.20 | 0.004 | Yes | 0.21 | 38 |
| | Nov 13-20 | 0.67 | 0.16 | <0.001 | Yes | 0.33 | 39 |

Key observations:

- **Middle-income relationship is weak when the huge outlier of the Seychelles is excluded and not statistically significant** in the November 2025 data (p = 0.105). A similiar estimate holds for Anthropic's previous data release. 

- Standard error for middle-income more than double between the two datasets (0.18 → 0.44)

- R² for middle-income fell from 14% to just 7%: in other words GDP per capita explains very little about relative AI adoption

## Repository Structure

```
anthropic-econ-critique/
├── README.md                    # This file
├── BLOG_POST.md                 # Full blog post
├── analysis_full.py             # Main Python analysis
├── analysis_results.csv         # Processed results
├── data/                        # Anthropic's original data
│   └── release_2026_01_15/      # November 2025 data (from Jan 2026 report)
├── original_report/             # Their report
└── figures/                     # Generated figures
```

## Reproducing Results

```bash
pip install pandas numpy statsmodels scipy
python analysis_full.py
```

## Data Source

Original data from Anthropic's HuggingFace repository:
- https://huggingface.co/datasets/Anthropic/EconomicIndex

## Contact

- Ilan Strauss: ilan@aidisclosures.org
- AI Disclosures Project: https://aidisclosures.org
