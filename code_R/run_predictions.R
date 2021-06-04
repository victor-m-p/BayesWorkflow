#' ---
#' title: "run_predictions"
#' output: html_document
#' ---
#' 
#' setup
#' 
## -----------------------------------------------------------------------------
# working directory 
#setwd("~/BayesWorkflow/code_r")

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
m_pooled_posterior <- readRDS("../models_R/m_multilevel_generic_fit.rds")


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
  fit = m_pooled_posterior,
  title = "R/brms: Predictions intervals (.95, .8 HDI) unseen data",
  data = test
)

save_plot(
  path = "../plots_R/multilevel_generic_HDI_predictions.png"
)


#' 
#' 
