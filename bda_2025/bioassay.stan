// Bioassay logistic regression model
data {
  int<lower=0> N;           // Number of data points
  vector[N] x;              // Dose (covariate)
  array[N] int<lower=0> n;  // Number of animals (trials)
  array[N] int<lower=0> y;  // Number of deaths (successes)
}

parameters {
  real alpha;  // Intercept
  real beta;   // Slope
}

model {
  // Priors
  alpha ~ normal(0, 2);
  beta ~ normal(10, 10);
  
  // Likelihood
  // This is the binomial GLM with a logit link
  y ~ binomial_logit(n, alpha + beta * x);
}

generated quantities {
  // Calculate LD50, the dose at which prob = 0.5
  // logit(0.5) = 0 = alpha + beta * x  =>  x = -alpha / beta
  real ld50 = -alpha / beta;
}