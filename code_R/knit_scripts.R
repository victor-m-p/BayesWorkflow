# knit .Rmd to .R to run whole pipeline from .sh script. 
library(knitr)
purl("run_data.Rmd", output = "run_data.R", documentation = 2)
purl("run_pooled.Rmd", output = "run_pooled.R", documentation = 2)
purl("run_multilevel.Rmd", output = "run_multilevel.R", documentation = 2)
purl("run_student.Rmd", output = "run_student.R", documentation = 2)
purl("run_model_comparison.Rmd", output = "run_model_comparison.R", documentation = 2)
purl("run_predictions.Rmd", output = "run_predictions.R", documentation = 2)
