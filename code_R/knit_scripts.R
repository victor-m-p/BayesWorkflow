# knit .Rmd to .R to run whole pipeline from .sh script.
library(knitr)
purl("run_data.Rmd", output = "run_data.R", documentation = 2)
purl("run_pooled.Rmd", output = "run_pooled.R", documentation = 2)
purl("run_intercept.Rmd", output = "run_intercept.R", documentation = 2)
purl("run_covariation.Rmd", output = "run_covariation.R", documentation = 2)
purl("run_model_comparison.Rmd", output = "run_model_comparison.R", documentation = 2)
purl("run_predictions.Rmd", output = "run_predictions.R", documentation = 2)