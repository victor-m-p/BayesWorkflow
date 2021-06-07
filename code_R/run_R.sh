#!/bin/sh
Rscript knit_scripts.R
Rscript run_data.R
Rscript run_pooled.R
Rscript run_multilevel.R
Rscript run_student.R
Rscript run_model_comparison.R
Rscript run_predictions.R
