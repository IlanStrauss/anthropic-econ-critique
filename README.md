# Does Anthropic's Economic Index really show growing international divergence in AI adoption and impacts?
# A Critique

**Authors**: Ilan Strauss
**AI Disclosures Project** (v.2)

---

## Summary: Anthropic's Claims vs. Our Critique

| Anthropic's Claim | Critique | Our Evidence |
|-------------------|--------------|--------------|
| "Uneven adoption remains well-explained by GDP per capita" (global β=0.71) | GDP per capita explains little of middle-income country adoption | Middle-income β=0.44, R²=17% (excl. Seychelles outlier) |
| AI usage concentration "essentially unchanged" (Aug–Nov 2025) | Gini captures a static snapshot; differential growth rates compound into widening gaps | Adoption growth rates show inequality: High-income +26%, Middle-income +14%, Low-income +22% |
| "Divergence in living standards" possible (McCrory to FT) | Speculative: no second-order effects estimated (AI → productivity → growth) | Cross-country data is Claude front-end (closer to consumer) not API (closer to firm-level)|
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

Anthropic's [Economic Index January 2026 Report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) claims "Worldwide, uneven [AI] adoption remains well-explained by GDP per capita." 

The [*Financial Times*](https://www.ft.com/content/3ad44e30-c738-4356-91fb-8bb2368685c4) covered this with Anthropic's head of economics warning: "If the productivity gains...materialise in places that have early adoption, you could see a divergence in living standards."

**Using their own data, we show that a country's income level (GDP per capita) does not meaningfully predict AI adoption for middle-income countries — where much of the world's population resides.** Moreover, no evidence on "productivity gains" is provided in Anthropic's report.

Using Anthropic's own data (both the November 2025 data from their January 2026 report and an earlier August 2025 release) we estimate separate regression estimates for different country groups and find that:

1. **For middle-income countries, GDP does not meaningfully predict AI adoption** — in the November 2025 data (used in their report), the relationship is not statistically meaningful (β = 0.73, SE = 0.44, p = 0.105); in the August 2025 data, the relationship is weak (β = 0.44, SE = 0.18, p = 0.019). The relationship is weak and highly uncertain — GDP per capita explains only 7-14% of the variation (R²), and the coefficient swings dramatically between periods (0.44 to 0.73) while remaining statistically weak or insignificant (N = 38 countries in each period)
2. **Middle-income countries adopt AI beyond what their income predicts** — education, infrastructure, and policy may drive adoption more than income
3. **No evidence supporting divergence is provided** — their single cross-sectional estimate does not imply growing divergence in AI adoption between countries (or resulting GDP growth). If anything, for middle-income countries, the data supports convergence in AI adoption.

The real question of divergence in economic growth between countries arising from AI investment requires a separate analysis.

![Their View vs Our View](figures/fig1_their_view_vs_ours.png)

![Uncertainty by Income Group](figures/fig2_uncertainty_by_group.png)

## Key Findings

### Weak and Highly Uncertain with Little Explanatory Power

Using two data releases from Anthropic (August 4-11 and November 13-20, 2025), *we show that the GDP-AI adoption relationship is weak and highly uncertain in Anthropic's datasets**, across both time periods of data:

| Income Group | Period | Slope (β) | Std. Error | p-value | Significant? | R² | N |
|--------------|--------|-----------|------------|---------|--------------|-----|---|
| **Global** | Aug 4-11 | 0.69 | 0.04 | <0.001 | Yes | 0.71 | 114 |
| | Nov 13-20 | 0.71 | 0.06 | <0.001 | Yes | 0.56 | 116 |
| **Low income** | Aug 4-11 | 0.76 | 0.19 | <0.001 | Yes | 0.30 | 38 |
| | Nov 13-20 | 0.85 | 0.18 | <0.001 | Yes | 0.37 | 39 |
| **Middle income** | Aug 4-11 | 0.44 | 0.18 | 0.019 | Yes | 0.14 | 38 |
| | **Nov 13-20 (with Seychelles)** | **0.73** | **0.44** | **0.105** | **No** | **0.07** | 38 |
| | Nov 13-20 (excl. Seychelles) | 0.44 | 0.16 | 0.011 | Yes | 0.17 | 37 |
| **High income** | Aug 4-11 | 0.63 | 0.20 | 0.004 | Yes | 0.21 | 38 |
| | Nov 13-20 | 0.67 | 0.16 | <0.001 | Yes | 0.33 | 39 |

Key observations:
- **Middle-income relationship is NOT statistically significant** in the November 2025 data (p = 0.105)
- Standard error for middle-income more than double between the two datasets (0.18 → 0.44)
- R² for middle-income fell from 14% to just 7% — GDP per capita explains very little abotu relative AI adoption
- Sample composition barely changes though (36 of 38 countries same in both periods)

## Why This Matters: Policy Implications

Anthropic's analysis implies a simple story: **higher country income level → more AI adoption**. Our findings show this story does not hold for middle-income countries.

### Their Story vs. Our Story

| Anthropic's Claim | Our Finding | Implication |
|-------------------|-------------|-------------|
| "Worldwide, uneven [AI] adoption remains well-explained by GDP per capita" | Not true for middle-income countries — relationship is not statistically significant (p = 0.105) | Most of the world's population does not fit their story |
| Single global elasticity of 0.71 | For middle-income countries, GDP explains only 7% of variation (R² = 0.07) | One-size-fits-all policy is inappropriate |
| Stable relationship | Coefficients and R² swing dramatically between August and November 2025 | Cross-sectional snapshots are unreliable for policy |

### What Should Policymakers Know?

**For low-income countries (β = 0.85, p < 0.001):**
- Income level *is* strongly associated with AI adoption
- Economic development may be a necessary precondition
- Possible focus: basic infrastructure, connectivity, economic fundamentals

**For middle-income countries (with Seychelles: β = 0.73, p = 0.105; excl. Seychelles: β = 0.44, p = 0.011):**
- **GDP does not significantly predict AI adoption** when including the Seychelles VPN outlier
- Even excluding Seychelles, the relationship is weak (β = 0.44 vs global 0.71, R² = 17%)
- Education, English proficiency, digital literacy, tech infrastructure, and regulatory environment may be driving adoption instead
- Examples: Brazil, Mexico, Thailand, Malaysia don't need to wait to get richer

**For high-income countries (β = 0.67, p < 0.001):**
- Significant relationship, but still substantial variation unexplained by income
- Israel is a notable over-adopter; Gulf states are under-adopters despite wealth
- Cultural, linguistic, and policy factors may dominate

### The Middle-Income Story

Our most striking finding: for middle-income countries, GDP per capita does not significantly predict AI adoption. The relationship is only significant at the 10% level (p = 0.105) and GDP explains only 7% of the variation (R² = 0.07).

This contrasts with low-income countries (where income level does strongly predict adoption) and high-income countries (where the relationship is also significant). Middle-income nations appear to be finding other pathways to AI adoption.

This is potentially good news: these countries may not need to wait to get richer. Education, infrastructure, and policy could drive adoption now — though this requires further investigation.

### No Evidence Supporting Divergence Is Provided

Anthropic warns of "divergence in living standards." But a cross-sectional income-AI elasticity does not imply divergence. Divergence requires a dynamic feedback loop: GDP growth → AI adoption growth → productivity growth → more GDP growth. Anthropic only estimates the first link (income levels → AI adoption levels). The divergence claim is speculation beyond their data.

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
