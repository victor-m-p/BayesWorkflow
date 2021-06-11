#' ---
#' title: "run_predictions"
#' output: html_document
#' ---
#' 
#' setup
#' 
## -----------------------------------------------------------------------------

# packages
pacman::p_load(tidyverse, 
               brms,
               modelr,
               tidybayes,
               bayesplot)

# load functions from fun_models.R
source("fun_models.R")
source("fun_helper.R")

#' 
#' load the model
#' 
## -----------------------------------------------------------------------------

# sampled models
m_covariation_posterior <- readRDS("../models_R/m_covariation_generic_fit.rds")


#' 
#' load test data
#' 
## -----------------------------------------------------------------------------

# read test data
test <- read_csv("../data/test.csv") %>%
  mutate(idx = as_factor(idx))


#' 
#' generate and plot predictions
#' 
## -----------------------------------------------------------------------------

plot_predicted_groups(
  fit = m_covariation_posterior,
  title = "R/brms: Predictions intervals (predictions)",
  data = test
)

save_plot(
  path = "../plots_R/covariation_generic_HDI_predictions.png"
)


#' 
#' 
