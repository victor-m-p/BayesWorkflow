#' ---
#' title: "Untitled"
#' output: html_document
#' ---
#' set-up
#' 
## -----------------------------------------------------------------------------

# packages
pacman::p_load(tidyverse, 
               brms,
               modelr,
               tidybayes,
               bayesplot)

RANDOM_SEED = 42

# load functions from fun_models.R
source("fun_models.R")
source("fun_helper.R")


#' 
#' global arguments
#' 
## -----------------------------------------------------------------------------

# Only this is flexible at the moment. Not as nice as the python setup. 
n_pp = 100


#' 
#' load data and ensure correct format.
#' 
## -----------------------------------------------------------------------------

# load data
train <- read_csv("../data/train.csv") %>%
  mutate(idx = as_factor(idx))


#' 
#' specify formula. 
#' 
## -----------------------------------------------------------------------------

#formula 
f_student <- bf(y ~ 1 + t + (1+t|idx)) # complete pooling 


#' 
#' Get priors and set priors. 
#' 
## -----------------------------------------------------------------------------

# fit the first model
get_prior(formula = f_student,
          data = train,
          family = student)

# set priors: three levels.
prior_student_specific <- c(
  prior(normal(0, 0.05), class = b),
  prior(normal(1.5, 0.05), class = Intercept),
  prior(normal(0, 0.05), class = sd),
  prior(normal(0, 0.05), class = sigma),
  prior(lkj(1), class = cor),
  prior(gamma(2, 0.1), class = nu) # same as in python. 
)

prior_student_generic <- c(
  prior(normal(0, 0.5), class = b),
  prior(normal(1.5, 0.5), class = Intercept),
  prior(normal(0, 0.5), class = sd),
  prior(normal(0, 0.5), class = sigma),
  prior(lkj(1), class = cor),
  prior(gamma(2, 0.1), class = nu)
)

prior_student_weak <- c(
  prior(normal(0, 5), class = b),
  prior(normal(1.5, 5), class = Intercept),
  prior(normal(0, 5), class = sd),
  prior(normal(0, 5), class = sigma),
  prior(lkj(1), class = cor),
  prior(gamma(2, 0.1), class = nu)
)


#' 
#' Now we compile models and check prior predictive.
#' 
## -----------------------------------------------------------------------------

# compile the models
m_student_specific_prior <- fit_mod(
  formula = f_student,
  family = student,
  data = train,
  prior = prior_student_specific,
  sample_prior = "only",
  file = "../models_R/m_student_specific_prior",
  random_seed = RANDOM_SEED
)

m_student_generic_prior <- fit_mod(
  formula = f_student,
  family = student,
  data = train,
  prior = prior_student_generic,
  sample_prior = "only",
  file = "../models_R/m_student_generic_prior",
  random_seed = RANDOM_SEED
)

m_student_weak_prior <- fit_mod(
  formula = f_student,
  family = student,
  data = train,
  prior = prior_student_weak,
  sample_prior = "only",
  file = "../models_R/m_student_weak_prior",
  random_seed = RANDOM_SEED
)


#' 
#' check the prior predictive checks
#' 
## -----------------------------------------------------------------------------

# specific model 
student_specific_prior_pred <- pp_check(m_student_specific_prior, 
                                     nsamples = 100) +
  labs(title = "R/brms: prior predictive check") 

save_plot(path = "../plots_R/student_specific_prior_pred.png")

# generic model
student_generic_prior_pred <- pp_check(m_student_generic_prior, 
                                         nsamples = 100) + 
  labs(title = "R/brms: prior predictive check")

save_plot(path = "../plots_R/student_generic_prior_pred.png")

# weak model
student_weak_prior_pred <- pp_check(m_student_weak_prior, 
                                    nsamples = 100) + 
  labs(title = "R/brms: prior predictive check")

save_plot(path = "../plots_R/student_weak_prior_pred.png")


#' 
#' fit the models 
#' 
## -----------------------------------------------------------------------------

# fit the models
m_student_specific_fit <- fit_mod(
  formula = f_student,
  family = student,
  data = train,
  prior = prior_student_specific,
  sample_prior = TRUE,
  file = "../models_R/m_student_specific_fit",
  random_seed = RANDOM_SEED
)

m_student_generic_fit <- fit_mod(
  formula = f_student,
  family = student,
  data = train,
  prior = prior_student_generic,
  sample_prior = TRUE,
  file = "../models_R/m_student_generic_fit",
  random_seed = RANDOM_SEED
)

m_student_weak_fit <- fit_mod(
  formula = f_student,
  family = student,
  data = train,
  prior = prior_student_weak,
  sample_prior = TRUE,
  file = "../models_R/m_student_weak_fit",
  random_seed = RANDOM_SEED
)


#' 
#' plot trace
#' 
## -----------------------------------------------------------------------------

# specific model
save_chains(
  fit = m_student_specific_fit,
  path = "../plots_R/student_specific_plot_trace.png"
)

# generic model
save_chains(
  fit = m_student_generic_fit,
  path = "../plots_R/student_generic_plot_trace.png"
)

# weak model
save_chains(
  fit = m_student_weak_fit,
  path = "../plots_R/student_weak_plot_trace.png"
)


#' 
#' posterior predictive checks 
#' 
## -----------------------------------------------------------------------------

# specific model 
student_specific_posterior_pred <- pp_check(m_student_specific_fit, 
                                         nsamples = 100) + 
  labs(title = "R/brms: posterior predictive check") 

save_plot(path = "../plots_R/student_specific_posterior_pred.png")

# generic model
student_generic_posterior_pred <- pp_check(m_student_generic_fit, 
                                         nsamples = 100) +
  labs(title = "R/brms: posterior predictive check")

save_plot(path = "../plots_R/student_generic_posterior_pred.png")

# weak model
student_weak_posterior_pred <- pp_check(m_student_weak_fit, 
                                    nsamples = 100) + 
  labs(title = "R/brms: posterior predictive check")

save_plot(path = "../plots_R/student_weak_posterior_pred.png")


#' 
#' ### HDI vs data ### 
#' 
#' fixed effects HDI vs. data
#' 
## -----------------------------------------------------------------------------

# specific
fixed_interval_groups(fit = m_student_specific_fit,
                    title = "R/brms: Prediction intervals (fixed)",
                    data = train,
                    n_time = 100)

save_plot(path = "../plots_R/student_specific_HDI_fixed.png")

# generic
fixed_interval_groups(fit = m_student_generic_fit,
                    title = "R/brms: Prediction intervals (fixed)",
                    data = train,
                    n_time = 100)

save_plot(path = "../plots_R/student_generic_HDI_fixed.png")

# weak
fixed_interval_groups(fit = m_student_weak_fit,
                    title = "R/brms: Prediction intervals (fixed)",
                    data = train,
                    n_time = 100)

save_plot(path = "../plots_R/student_weak_HDI_fixed.png")


#' 
#' prediction intervals
#' 
## -----------------------------------------------------------------------------

# specific
prediction_interval_groups(fit = m_student_specific_fit, 
                         title = "R/brms: Prediction intervals (full)",
                         data = train,
                         n_time = 100)

save_plot(path = "../plots_R/student_specific_HDI_full.png")

# generic
prediction_interval_groups(fit = m_student_generic_fit, 
                         title = "R/brms: Prediction intervals (full)",
                         data = train,
                         n_time = 100)

save_plot(path = "../plots_R/student_generic_HDI_full.png")

# weak
prediction_interval_groups(fit = m_student_weak_fit, 
                         title = "R/brms: Prediction intervals (full)",
                         data = train,
                         n_time = 100)

save_plot(path = "../plots_R/student_weak_HDI_full.png")


#' 
#' ### MCMC (hdi for parameters) ###
#' 
## -----------------------------------------------------------------------------

# for consistency with python. 
# does not do anything at the moment..
width = 7
height = 4

# specific
mcmc_hdi(fit = m_student_specific_fit, 
         title = "R/brms: HDI intervals for parameters")

save_plot(path = "../plots_R/student_specific_HDI_param.png",
          width = width,
          height = height)

# generic
mcmc_hdi(fit = m_student_generic_fit, 
         title = "R/brms: HDI intervals for parameters")

save_plot(path = "../plots_R/student_generic_HDI_param.png",
          width = width,
          height = height)

# weak
mcmc_hdi(fit = m_student_weak_fit, 
         title = "R/brms: HDI intervals for parameters")

save_plot(path = "../plots_R/student_weak_HDI_param.png",
          width = width,
          height = height)


#' 
#' 
