# function for fitting models. 
fit_mod <- function(formula,
                    family,
                    data,
                    prior,
                    sample_prior,
                    file){
  
  b <- brm(
    formula = formula,
    family = family,
    data = data,
    prior = prior,
    cores = 4, chains = 2, 
    sample_prior = sample_prior,
    backend = "cmdstanr",
    file = file,
    file_refit = "on_change",
    threads = threading(2),
    control = list(adapt_delta = .99, # perhaps just .95
                   max_treedepth = 20)
  )
  
  return(b)
  
}
