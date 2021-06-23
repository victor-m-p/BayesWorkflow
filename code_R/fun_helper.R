
### save plots in consistent format ###
save_plot <- function(path, height = 7, width = 10){
  ggsave(path,
         units = "in",
         height = 7,
         width = 10)
}

#### check chains and save output ###
save_chains <- function(fit, path){
  png(filename=path,
      width = 1500, height = 1500)
  plot(fit,
       N = 10) # testing this. 
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
                    color = "#08519C") + # hdi interval?
    geom_jitter(data = data, 
                color = "navyblue", 
                shape = 1, 
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() +
    theme_minimal() +
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
                    color = "#08519C") + # hdi interval?
    geom_jitter(data = data, 
                color = "navyblue", 
                shape = 1,
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() + 
    theme_minimal() + 
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
    theme_minimal() + 
    ggtitle(title)
}

## grouped (multilevel, student) ##
prediction_interval_groups <- function(fit, title, data = train, n_time = 100){
  
  data %>%
    data_grid(t = seq_range(t, n = n_time), idx) %>%
    add_predicted_draws(fit,
                        allow_new_levels = T) %>%
    ggplot(aes(x = t, y = y)) + 
    stat_lineribbon(aes(y = .prediction), 
                    .width = c(.95, .8), 
                    color = "#08519C") + #hdi interval?
    geom_jitter(data = data, 
                color = "navyblue", 
                shape = 1, 
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() +
    theme_minimal() + 
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
                    color = "#08519C") + #hdi interval?
    geom_jitter(data = data, 
                color = "navyblue", 
                shape = 1,
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() +
    theme_minimal() + 
    ggtitle(title)
  
}

### MCMC (hdi for parameters) ###
mcmc_hdi <- function(fit, title){
  
  mcmc_areas(
    fit,
    pars = c("b_Intercept",
             "b_t",
             "sigma"),
    prob = 0.8, # 80% intervals
    prob_outer = 0.99, # 99%
    point_est = "mean") + 
    theme_minimal() + 
    ggtitle(title)
  
}


# idx predictions 
hdi_ID <- function(fit, data = train, ID){
  
  data %>% 
    data_grid(t = t, idx = ID) %>%
    add_predicted_draws(fit,
                        allow_new_levels = T) %>%
    ggplot(aes(x = t, y = y)) + 
    stat_interval(aes(y = .prediction), .width = c(.95, .8)) + 
    stat_slab(aes(y = .prediction), 
              .width = c(.95, .8), 
              position = position_nudge(x = 0.1),
              fill = "#08519C",
              alpha = 0.3) +
    geom_point(data = data %>% filter(idx == ID),
               color = "navyblue",
               size = 2) +
    scale_x_continuous(breaks = seq(0, 9, 1)) + 
    scale_color_brewer() +
    theme_minimal() +
    ggtitle(glue("R/brms: HDI prediction intervals (Alien {ID})"))
  
}
