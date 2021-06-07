# function for fitting models. 
fit_mod <- function(formula,
                    family,
                    data,
                    prior,
                    sample_prior,
                    file, 
                    random_seed){
  
  b <- brm(
    formula = formula,
    family = family,
    data = data,
    prior = prior,
    sample_prior = sample_prior,
    cores = 4, chains = 2, 
    backend = "cmdstanr",
    iter = 4000, warmup = 2000, # consistency with python
    file = file,
    file_refit = "on_change",
    threads = threading(2),
    seed = random_seed,
    control = list(adapt_delta = .99, ### CHANGE
                   max_treedepth = 20) ### CHANGE
  )
  
  return(b)
  
}

