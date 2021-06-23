#' ---
#' title: "Untitled"
#' output: html_document
#' ---
#' 
#' 
#' setup
#' 
## -----------------------------------------------------------------------------

# packages
pacman::p_load(tidyverse, 
               brms,
               modelr,
               tidybayes,
               bayesplot,
               glue)

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

prediction_interval_groups(
  fit = m_covariation_posterior,
  title = "R/brms: Predictions intervals (predictions)",
  data = test
)

save_plot(
  path = "../plots_R/covariation_generic_HDI_predictions.png"
)


#' 
#' idx predictions
#' 
## -----------------------------------------------------------------------------

# get unique idx
idx_unique <- as.numeric(unique(test$idx)) - 1

# loop over idx and use function
for (ID in idx_unique){
  
  # generate plot
  hdi_ID(fit = m_covariation_posterior,
         data = test,
         ID = ID)
  
  # save plot
  save_plot(path = glue("../plots_R/covariation_generic_HDI_individual{ID}_test.png"))
}


#' 
