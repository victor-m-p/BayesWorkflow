
### save plots in consistent format ###
save_plot <- function(path, height = 7, width = 10){
  ggsave(path,
         units = "in",
         height = 7,
         width = 10)
}

#### check chains and save output ###
save_chains <- function(fit, path){
  png(filename=path)
  plot(fit,
       N = 10) 
  dev.off()
}

### Visualize models ###

## pooled ##

# prediction intervals for pooled model. 
prediction_interval_pool <- function(fit, title, data = train, n_time = 100){
  
  data %>%
    data_grid(t = seq_range(t, n = n_time)) %>%
    add_predicted_draws(fit) %>%
    ggplot(aes(x = t, y = y)) +
    stat_lineribbon(aes(y = .prediction), 
                    .width = c(.95, .8),
                    color = "#08519C") +
    geom_jitter(data = data, 
                color = "navyblue", 
                shape = 1, 
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() +
    ggtitle(title)
  
}

# posterior fit intervals for pooled model.
fixed_interval_pool <- function(fit, title, data = train, n_time = 100){
  
  data %>%
    data_grid(t = seq_range(t, n = n_time)) %>%
    add_fitted_draws(fit) %>%
    ggplot(aes(x = t, y = y)) + 
    stat_lineribbon(aes(y = .value), 
                    .width = c(.95, .8), 
                    color = "#08519C") +
    geom_jitter(data = data, 
                color = "navyblue", 
                shape = 1,
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() + 
    ggtitle(title)
  
}

# Kruschke style. 
fixed_kruschke_pool <- function(fit, title, data = train, n_time = 100){
  
  data %>% 
    data_grid(t = seq_range(t, n = n_time)) %>%
    add_fitted_draws(fit, n = 100) %>%
    ggplot(aes(x = t, y = y)) + 
    geom_jitter(data = train, 
                color = "navyblue",
                shape = 1,
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() + 
    geom_line(aes(y = .value, group = .draw), 
              alpha = 1/20, 
              color = "#08519C") +
    ggtitle(title)
}

## grouped (multilevel, student) ##
prediction_interval_groups <- function(fit, title, data = train, n_time = 100){
  
  data %>%
    data_grid(t = seq_range(t, n = n_time), idx) %>%
    add_predicted_draws(fit) %>%
    ggplot(aes(x = t, y = y)) + 
    stat_lineribbon(aes(y = .prediction), 
                    .width = c(.95, .8), 
                    color = "#08519C") +
    geom_jitter(data = data, 
                color = "navyblue", 
                shape = 1, 
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() +
    ggtitle(title)
  
}

fixed_interval_groups <- function(fit, title, data = train, n_time = 100){
  
  data %>%
    data_grid(t = seq_range(t, n = n_time), idx) %>%
    add_fitted_draws(fit,
                     re_formula = NA) %>%
    ggplot(aes(x = t, y = y)) + 
    stat_lineribbon(aes(y = .value), 
                    .width = c(.95, .8), 
                    color = "#08519C") +
    geom_jitter(data = data, 
                color = "navyblue", 
                shape = 1,
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() +
    ggtitle(title)
  
}

### MCMC (hdi for parameters) ###
mcmc_hdi <- function(fit, title){
  
  mcmc_plot(fit,
            pars = c("b_Intercept",
                     "b_t",
                     "sigma"),
            fixed = T) +
    ggtitle(title)
  
}

### Predictions on unseen data ###

# only for multilevel (because it is best). 
plot_predicted_groups <- function(fit, title, data = train, n_time = 100){
  
  data %>%
    data_grid(t = seq_range(t, n = n_time), idx) %>%
    add_predicted_draws(fit) %>%
    ggplot(aes(x = t, y = y)) + 
    stat_lineribbon(aes(y = .prediction), 
                    .width = c(.95, .8), 
                    color = "#08519C") +
    geom_jitter(data = data, 
                color = "navyblue", 
                shape = 1, 
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() +
    ggtitle(title)
  
}

