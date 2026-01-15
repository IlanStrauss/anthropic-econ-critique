# =============================================================
# Critique of Anthropic Economic Index: Partial Pooling with brms
# =============================================================
# Uses brms (Paul Bürkner's package) for Bayesian hierarchical models
# Can use ADVI for fast approximate inference if MCMC is too slow

library(brms)
library(tidyverse)
library(broom)

# Load data
df <- read_csv("analysis_results.csv")

# Quick look at data
cat("\n=== Data Summary ===\n")
cat("Countries:", nrow(df), "\n")
print(df %>% select(geo_id, geo_name, log_gdp, log_usage, income_tercile) %>% head(10))

# =============================================================
# 1. Reproduce OLS
# =============================================================

cat("\n=== OLS (Reproducing Anthropic) ===\n")
ols_model <- lm(log_usage ~ log_gdp, data = df)
print(summary(ols_model))
cat("OLS Slope:", coef(ols_model)["log_gdp"], "\n")

# =============================================================
# 2. Partial Pooling: Random Intercepts
# =============================================================

cat("\n=== Partial Pooling: Random Intercepts ===\n")
cat("Model: log_usage ~ log_gdp + (1 | income_tercile)\n")

# Using ADVI for speed (variational approximation)
# To use full MCMC, remove algorithm = "meanfield"
ri_model <- brm(
  log_usage ~ log_gdp + (1 | income_tercile),
  data = df,
  algorithm = "meanfield",  # Fast VI approximation
  iter = 20000,
  seed = 42
)

print(summary(ri_model))
cat("\nFixed effect (log_gdp slope):\n")
print(fixef(ri_model))

# =============================================================
# 3. Partial Pooling: Random Slopes
# =============================================================

cat("\n=== Partial Pooling: Random Slopes ===\n")
cat("Model: log_usage ~ log_gdp + (1 + log_gdp | income_tercile)\n")

rs_model <- brm(
  log_usage ~ log_gdp + (1 + log_gdp | income_tercile),
  data = df,
  algorithm = "meanfield",  # Fast VI approximation
  iter = 20000,
  seed = 42
)

print(summary(rs_model))
cat("\nFixed effects:\n")
print(fixef(rs_model))

cat("\nGroup-specific slopes:\n")
print(coef(rs_model)$income_tercile)

# =============================================================
# 4. Full MCMC (if you want proper uncertainty)
# =============================================================

# Uncomment to run full MCMC (slower but more accurate)
# rs_model_mcmc <- brm(
#   log_usage ~ log_gdp + (1 + log_gdp | income_tercile),
#   data = df,
#   chains = 4,
#   cores = 4,
#   iter = 4000,
#   warmup = 1000,
#   seed = 42,
#   control = list(adapt_delta = 0.95)
# )
# print(summary(rs_model_mcmc))

# =============================================================
# 5. Model Comparison
# =============================================================

cat("\n=== Effect Size Comparison ===\n")

comparison <- tibble(
  Method = c("OLS (Anthropic)", "Partial Pooling RI", "Partial Pooling RS"),
  Slope = c(
    coef(ols_model)["log_gdp"],
    fixef(ri_model)["log_gdp", "Estimate"],
    fixef(rs_model)["log_gdp", "Estimate"]
  ),
  SE = c(
    summary(ols_model)$coefficients["log_gdp", "Std. Error"],
    fixef(ri_model)["log_gdp", "Est.Error"],
    fixef(rs_model)["log_gdp", "Est.Error"]
  )
)

print(comparison)

cat("\nDifference (Partial Pooling - OLS):",
    fixef(rs_model)["log_gdp", "Estimate"] - coef(ols_model)["log_gdp"], "\n")
cat("% Change:",
    100 * (fixef(rs_model)["log_gdp", "Estimate"] - coef(ols_model)["log_gdp"]) / coef(ols_model)["log_gdp"],
    "%\n")

# =============================================================
# 6. Visualize Shrinkage
# =============================================================

# Get group-specific slopes
group_slopes <- coef(rs_model)$income_tercile[, , "log_gdp"]

# Unpooled OLS by group (for comparison)
unpooled <- df %>%
  group_by(income_tercile) %>%
  summarise(
    ols_slope = coef(lm(log_usage ~ log_gdp))[2],
    n = n()
  )

cat("\n=== Shrinkage Comparison ===\n")
cat("Unpooled (separate OLS per group):\n")
print(unpooled)
cat("\nPartial Pooling (shrunk toward global mean):\n")
print(group_slopes)

# =============================================================
# 7. Save Results
# =============================================================

saveRDS(list(
  ols = ols_model,
  ri = ri_model,
  rs = rs_model,
  comparison = comparison
), "brms_results.rds")

cat("\n=== Analysis Complete ===\n")
cat("Results saved to brms_results.rds\n")
