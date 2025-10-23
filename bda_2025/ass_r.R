# Load required libraries
library(cmdstanr)
install_cmdstan()
library(posterior)
library(bayesplot)

# --- 1. Compile the Stan Model ---
# Make sure "bioassay.stan" is in your working directory
# or provide the full path
tryCatch({
  mod <- cmdstan_model("bioassay.stan")
}, error = function(e) {
  message("Make sure 'bioassay.stan' is in your R working directory.")
  stop(e)
})

# --- 2. Define the Data ---
# Data from BDA3, p. 77
x <- c(-0.86, -0.30, -0.05, 0.73)
n <- c(5, 5, 5, 5)
y <- c(0, 1, 3, 5)

# Put data into a list for Stan
data_list <- list(N = 4, x = x, n = n, y = y)

# --- 3. Run the Sampler ---
fit <- mod$sample(
  data = data_list,
  seed = 4911,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000, # This results in 2000 total draws per chain
  parallel_chains = 4,
  refresh = 0 # Suppress printing progress
)

# --- 4. Check Diagnostics and Get Answers ---

# 4.3: Check for warnings
message("--- 4.3: Diagnostic Summary ---")
fit$diagnostic_summary()
# This shows 0 divergences and 0 max_treedepth hits.
# Answer: No divergences or max_treedepths reached

# 4.4 - 4.9: Get Rhat and ESS
# Use summarize_draws to get all diagnostics at once
draws_df <- fit$draws(format = "df")
diags <- summarize_draws(
  draws_df,
  "rhat_basic",
  "ess_mean",
  ess_q05 = ~ess_quantile(.x, 0.05)
)

message("\n--- 4.4 - 4.9: Rhat and ESS values ---")
print(diags)

# 4.4: Rhat for alpha
alpha_rhat <- diags[diags$variable == "alpha", "rhat_basic"]
message("\n4.4 Rhat for alpha: ", round(alpha_rhat, 3))

# 4.5: Rhat for beta
beta_rhat <- diags[diags$variable == "beta", "rhat_basic"]
message("4.5 Rhat for beta: ", round(beta_rhat, 3))

# 4.6: ESS mean for alpha
alpha_ess_mean <- diags[diags$variable == "alpha", "ess_mean"]
message("4.6 ESS mean for alpha: ", round(alpha_ess_mean))

# 4.7: ESS mean for beta
beta_ess_mean <- diags[diags$variable == "beta", "ess_mean"]
message("4.7 ESS mean for beta: ", round(beta_ess_mean))

# 4.8: ESS q0.05 for alpha
alpha_ess_q05 <- diags[diags$variable == "alpha", "ess_q05"]
message("4.8 ESS q0.05 for alpha: ", round(alpha_ess_q05))

# 4.9: ESS q0.05 for beta
beta_ess_q05 <- diags[diags$variable == "beta", "ess_q05"]
message("4.9 ESS q0.05 for beta: ", round(beta_ess_q05))


# 4.7 (second question): Plot ACF
message("\n--- 4.7 (second question): Plotting ACF ---")
# The plot will show ACFs dropping immediately to 0,
# which is much faster than a typical MH algorithm for this problem.
acf_plot <- mcmc_acf(fit$draws(), pars = c("alpha", "beta"))
print(acf_plot)
