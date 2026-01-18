# =============================================================
# Critique of Anthropic Economic Index: Partial Pooling with brms
# =============================================================
# Bayesian hierarchical model with 7 country groups
# Produces: β = 0.54, SE = 0.10, 95% CI = [0.33, 0.74]

library(brms)
library(tidyverse)

# Load data
df <- read_csv("analysis_results.csv")

cat("\n=== Data Summary ===\n")
cat("Countries:", nrow(df), "\n\n")

# =============================================================
# 1. Reproduce OLS (Anthropic's approach)
# =============================================================

cat("=== OLS (Reproducing Anthropic) ===\n")
ols_model <- lm(log_usage ~ log_gdp, data = df)
cat("OLS Slope:", round(coef(ols_model)["log_gdp"], 3), "\n")
cat("OLS SE:", round(summary(ols_model)$coefficients["log_gdp", "Std. Error"], 3), "\n")
cat("OLS 95% CI: [", round(confint(ols_model)["log_gdp", 1], 2), ", ",
    round(confint(ols_model)["log_gdp", 2], 2), "]\n\n")

# =============================================================
# 2. Create 7 theoretically meaningful groups
# =============================================================

# Define groups based on theoretical considerations:
# - Gulf: GCC oil states (high GDP, very low AI adoption)
# - TechHub: Known high AI adopters
# - LowAfrica: Low-income African countries
# - LowAsia: Low-income Asian countries
# - LowOther: Other low-income countries
# - Mid: Middle-income countries
# - High: High-income (excluding Gulf and TechHub)

gulf_countries <- c("QAT", "KWT", "SAU", "ARE", "BHR", "OMN")
techhub_countries <- c("ISR", "GEO", "ARM", "KOR", "MNE")

# African countries (by geo_id)
african_countries <- c("ZAF", "NGA", "KEN", "GHA", "TZA", "UGA", "ETH", "AGO",
                       "CIV", "CMR", "SEN", "ZWE", "ZMB", "MOZ", "BWA", "NAM",
                       "MUS", "RWA", "TUN", "MAR", "DZA", "EGY")

# Asian countries (by geo_id)
asian_countries <- c("IND", "IDN", "PHL", "VNM", "THA", "MYS", "BGD", "PAK",
                     "LKA", "NPL", "MMR", "KHM", "LAO", "MNG", "UZB", "KAZ",
                     "KGZ", "TJK", "AZE", "GEO", "ARM")

# Create income terciles first
df <- df %>%
  mutate(income_tercile = ntile(log_gdp, 3))

# Assign groups
df <- df %>%
  mutate(
    group7 = case_when(
      geo_id %in% gulf_countries ~ "Gulf",
      geo_id %in% techhub_countries ~ "TechHub",
      income_tercile == 1 & geo_id %in% african_countries ~ "LowAfrica",
      income_tercile == 1 & geo_id %in% asian_countries ~ "LowAsia",
      income_tercile == 1 ~ "LowOther",
      income_tercile == 2 ~ "Mid",
      income_tercile == 3 ~ "High",
      TRUE ~ "Mid"  # fallback
    )
  )

# Remove Gulf and TechHub from High if they ended up there
df <- df %>%
  mutate(
    group7 = case_when(
      geo_id %in% gulf_countries ~ "Gulf",
      geo_id %in% techhub_countries ~ "TechHub",
      TRUE ~ group7
    )
  )

cat("=== Group Sizes ===\n")
print(table(df$group7))
cat("\n")

# =============================================================
# 3. Unpooled OLS by group (for comparison)
# =============================================================

cat("=== Unpooled OLS by Group ===\n")
unpooled <- df %>%
  group_by(group7) %>%
  summarise(
    n = n(),
    ols_slope = coef(lm(log_usage ~ log_gdp))[2],
    ols_se = summary(lm(log_usage ~ log_gdp))$coefficients[2, 2],
    .groups = "drop"
  )
print(unpooled)
cat("\nSimple average of group slopes:", round(mean(unpooled$ols_slope), 2), "\n\n")

# =============================================================
# 4. Bayesian Hierarchical Model (MCMC)
# =============================================================

cat("=== Fitting Bayesian Hierarchical Model ===\n")
cat("Model: log_usage ~ log_gdp + (1 + log_gdp | group7)\n")
cat("Using MCMC with 4 chains, 4000 iterations...\n\n")

# Random intercepts and slopes by group
rs_model <- brm(
  log_usage ~ log_gdp + (1 + log_gdp | group7),
  data = df,
  chains = 4,
  cores = 4,
  iter = 4000,
  warmup = 1000,
  seed = 42,
  control = list(adapt_delta = 0.99)
)

# =============================================================
# 5. Results
# =============================================================

cat("\n=== Model Summary ===\n")
print(summary(rs_model))

cat("\n=== Fixed Effects (Global Estimates) ===\n")
fe <- fixef(rs_model)
print(fe)

global_slope <- fe["log_gdp", "Estimate"]
global_se <- fe["log_gdp", "Est.Error"]
global_ci_lower <- fe["log_gdp", "Q2.5"]
global_ci_upper <- fe["log_gdp", "Q97.5"]

cat("\n=== KEY RESULT ===\n")
cat("Global slope (β):", round(global_slope, 2), "\n")
cat("Standard Error:", round(global_se, 2), "\n")
cat("95% CI: [", round(global_ci_lower, 2), ", ", round(global_ci_upper, 2), "]\n")

cat("\n=== Comparison with Anthropic ===\n")
cat("Anthropic OLS: β = 0.69, SE = 0.04, CI = [0.61, 0.77]\n")
cat("Our estimate:  β =", round(global_slope, 2), ", SE =", round(global_se, 2),
    ", CI = [", round(global_ci_lower, 2), ",", round(global_ci_upper, 2), "]\n")

if (0.69 > global_ci_upper) {
  cat("\n** Anthropic's estimate (0.69) falls OUTSIDE our 95% CI **\n")
}

# =============================================================
# 6. Group-Specific Slopes (Partial Pooling)
# =============================================================

cat("\n=== Group-Specific Slopes (Partial Pooling) ===\n")
group_coefs <- coef(rs_model)$group7[, , "log_gdp"]
print(round(group_coefs, 2))

# Compare unpooled vs partial pooling
cat("\n=== Shrinkage Comparison ===\n")
comparison <- unpooled %>%
  left_join(
    tibble(
      group7 = rownames(group_coefs),
      pp_slope = group_coefs[, "Estimate"],
      pp_ci_lower = group_coefs[, "Q2.5"],
      pp_ci_upper = group_coefs[, "Q97.5"]
    ),
    by = "group7"
  ) %>%
  mutate(
    shrinkage = ols_slope - pp_slope
  )
print(comparison)

# =============================================================
# 7. Model Diagnostics
# =============================================================

cat("\n=== Model Diagnostics ===\n")
cat("Divergent transitions:", sum(nuts_params(rs_model)$Value[nuts_params(rs_model)$Parameter == "divergent__"]), "\n")
cat("Rhat values (should be ~1.00):\n")
print(rhat(rs_model))

# =============================================================
# 8. Save Results
# =============================================================

results <- list(
  ols = ols_model,
  brms = rs_model,
  groups = table(df$group7),
  global_slope = global_slope,
  global_se = global_se,
  global_ci = c(global_ci_lower, global_ci_upper),
  group_slopes = group_coefs,
  comparison = comparison
)

saveRDS(results, "brms_results.rds")
write_csv(df, "analysis_results_with_groups.csv")

cat("\n=== Analysis Complete ===\n")
cat("Results saved to brms_results.rds\n")
cat("Data with groups saved to analysis_results_with_groups.csv\n")
