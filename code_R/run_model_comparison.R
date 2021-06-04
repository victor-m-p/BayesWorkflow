#' ---
#' title: "Model Comparison (VMP)"
#' output: html_document
#' ---
#' 
#' set-up
#' 
## -----------------------------------------------------------------------------

# working directory 
#setwd("~/BayesWorkflow/code_r")

# packages
pacman::p_load(tidyverse, 
               brms,
               kableExtra, 
               magick, 
               webshot,
               modelr,
               tidybayes)

# load functions from fun_helper.R
source("fun_helper.R")


#' 
#' load data 
#' 
## -----------------------------------------------------------------------------

# load data
train <- read_csv("../data/train.csv")

# make sure that data format is okay
train <- train %>%
  mutate(idx = as_factor(idx))


#' 
#' load models 
#' 
## -----------------------------------------------------------------------------

# sampled models
m_pooled_posterior <- readRDS("../models_R/m_pooled_generic_fit.rds")
m_multilevel_posterior <- readRDS("../models_R/m_multilevel_generic_fit.rds")
m_student_posterior <- readRDS("../models_R/m_student_generic_fit.rds")


#' 
#' loo comparison (might not go in report). 
#' but we could try to get it to work with saving..
#' 
## -----------------------------------------------------------------------------

# add criterions 
m_pooled_posterior <- add_criterion(m_pooled_posterior, 
                                    criterion = c("loo", "bayes_R2"))

m_multilevel_posterior <- add_criterion(m_multilevel_posterior, 
                                        criterion = c("loo", "bayes_R2"))

m_student_posterior <- add_criterion(m_student_posterior, 
                                     criterion = c("loo", "bayes_R2")) # one problematic observation.

# run loo compare
loo_compare(m_pooled_posterior,
            m_multilevel_posterior,
            m_student_posterior)

# also uses stacking (and gives the same as pyMC3)
loo_model_weights(m_pooled_posterior,
                  m_multilevel_posterior,
                  m_student_posterior)


