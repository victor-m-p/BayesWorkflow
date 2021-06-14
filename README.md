# BayesWorkflow
Bayesian workflow in R (brms) and Python (PyMC3).
Made for the user who wants to transition from R (brms) to Python (pyMC3) or
the other way.
Originates as exam project for Cognitive Science Master's course Human
Computer Interaction (HCI).

# Streamlit
Link to **Streamlit** app: https://share.streamlit.io/victor-m-p/bayesworkflow/main/BayesWorkflow.py

The python document which runs the streamlit app is BayesWorkflow.py

# Workflow & Packages
This project (streamlit app) is meant to be used while coding
along in your own instance of R and/or Python.
To reproduce the analysis, simply navigate to the Streamlit app
and copy the code into your own editor while following.

It can be tricky to properly set up both *pyMC3* and *brms*
because they rely on interfacing with *theano* and *stan*.

For **Python** you will need to set up an environment with the
necessary packages installed (see **requirements.txt**).

For **R** you might need to set up "cmdstanr" and other
packages manually.

## Reproducibility: Running the whole pipeline
If you (for some reason) want to re-run the whole analysis
(to generate all figures and plots) this can be done in
**Python** from the /code\_python/run\_python.sh script and
in **R** from the /code\_R/run\_R.sh script.


