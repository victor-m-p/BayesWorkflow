# import packages
import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import arviz as az
import sklearn as sk

# we want this for all pages
st.set_page_config(
    page_title="Bayesian Workflow",
    page_icon="bayes_bois.png",
    layout="wide",
    initial_sidebar_state="expanded",
    )

st.image("../img/bayes_bois.png", width = 100)

# other layout options?
# 1. taking less space
# 2. perhaps not
choice = st.sidebar.radio('Sections:',["Dev", "Introduction", "Specification", "Validation", "Visualization", "Model Comparison"])


if choice == "Dev":
    '''
    # Current development/testing
    '''

    ## checking functionality:
    # https://docs.streamlit.io/en/stable/api.html

    '''
    # Coding functionality

    Checking coding functionality
    '''
    
    ## the formatting would be nice for individual words (like R)
    ## would also be nice with both show code & display output. 
    py_code1 = '''
    ## import libraries
    import pandas as pd 
    import seaborn as sns 
    
    ## read data
    d = pd.read_csv("../data/hh_budget.csv")

    ## explorative plotting 
    sns.lineplot(d, x = Year, y = Wealth, hue = Country)
    '''
    R_code1 = '''
    ## import libraries 
    pacman::p_load(tidyverse)
    
    ## read data 
    d = read_csv("../data/hh_budget.csv")
    
    ## explorative plotting 
    d %>% ggplot(aes(x = Year, y = Wealth, color = Country)) +
        geom_line()
    '''
    
    '''
    ## Option 1. 
    '''

    st.code(py_code1, language = 'Python')
    st.code(R_code1, language = 'R')
    
    '''
    ## Option 2. 
    '''

    col1, col2 = st.beta_columns(2)

    with col1:
        st.header("R")
        st.code(R_code1, language = "R")
    
    with col2: 
        st.header("Python")
        st.code(py_code1, language = "Python")
    
    '''
    # Expander functionality

    Checking expander functionality
    '''

    ## could be better, but pretty good already.
    expander = st.beta_expander("Deep Dive")
    expander.write(f"The reason why we do predictive posterior checks is ...")

    ## check st.write:
    '''
    # Checking basic functionality

    '''

    st.write("R") # just writes
    st.write(1234) # formats some things nicely.
    st.latex(r'''
            f(x, y) = x_{index} + y^{index}
            ''') # latex.


elif choice == "Introduction":

    """
    # Bayesian Workflow: from R to Python
    ### Victor Møller Poulsen

    This notebook aims to serve two purposes.

    1. Introduce a healthy workflow for bayesian modelling

    2. Introduce the reader coming from R/brms to Python/PyMC3

    (include a nice picture here)

    """

elif choice == "Specification":
    '''
    # Specifying a model

    This section will guide you through explorative visualization,
    and end with formulating priors for

    ## Explorative Visualization

    Before formulating models you will typically want to

    ## Formulating Models

    You will often want to do multiple models (corresponding to differ)

    ## Formulating Priors

    ## Monte Carlo Inference

    '''

elif choice == "Validation":
    '''
    # Validation

    ## Prior & Posterior predictive checks

    ## Updating Checks

    ## Checking the Chain(s)

    '''

elif choice == "Visualization":
    '''
    # Visualizing the model

    ## Types of plots

    e.g.

    ##

    '''

elif choice == "Model Comparison":
    '''


    '''
