# import packages
import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import arviz as az
import pickle
#import fun_models as fm
## @st.cache - use this?

# we want this for all pages
st.set_page_config(
    page_title="Bayesian Workflow",
    page_icon="bayes_bois.png",
    layout="wide",
    initial_sidebar_state="expanded",
    )

st.image("../img/bayes_bois.png", width = 100)

## basic preprocessing. 
## load data ##
with open('../data/train.pickle', 'rb') as f:
    train = pickle.load(f)

with open('../data/test.pickle', 'rb') as f:
    test = pickle.load(f)

## take out the arrays ##
t_train = train.t.values
idx_train = train.idx.values
y_train = train.y.values
n_train = len(np.unique(idx_train))

# other layout options?
# 1. taking less space
# 2. perhaps not
choice = st.sidebar.radio('Sections:', ["Introduction", "Simulation & EDA", "Complete Pooling (model 1)", "Multilevel (model 2)", "Student-t (model 3)", "Model Comparison", "Prediction", "Sandbox"])

if choice == "Introduction":
    '''
    # Purpose 
    
    something about what people should expect to get out of the
    
    notebook, and who the intended audience is. 
    
    Something I wish I had had when I started to transition from bayesian modeling 
    in R to Python. 
    
    '''
    
    '''
    # Progression
    
    
    ## Optimization of workflow
    
    As you will see, almost all the code used is the same
    for all three candidate models. 
    
    Typing all the code for each model is 
    redundant and a waste of typing. 
    
    In order to make the workflow effective we can create a 
    number of convenience functions. 
    
    We might also want to split up the analysis into several
    documents, such that we fit models in one document and
    explore them in others. 
    
    A non-exhaustive list of convenience functions that I 
    like to use are discussed in this section.
    
    Here we will also discuss how the workflow differs 
    between pyMC3 and brms. 
    
    '''
    
    '''
    # Functionality 
    
    You will see boxes with the titles:
    
    * "Code-Shark"
    * "Math-Whizz"
    * "Concept-Guru"
    
    These are for those interested in diving deeper into the material. 
    '''
    
    '''
    # Notes
    
    something on e.g. that model compilation and sampling typically
    
    being done separately in pymc3 and together in brms. 
    '''
    
    
elif choice == "Simulation & EDA":

    '''
    # Data (Simulation)
    For this notebook we will simulate our own data.
    
    The data will consist of ...
    
    If you are curious, check the "Code-Monkey" box. 
    '''
    
    # if people want to see the simulation code
    expander = st.beta_expander("Code-Monkey: simulation")
    
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1:
            st.write("Python code")
        with col2: 
            st.write("R code") 
            
    '''
    # Quick EDA
    Let's plot the data to get a feel. 
    '''
    col1, col2 = st.beta_columns(2)
    
    with col1: 
        st.image("../plots_python/p1.png")
        
    with col2: 
        st.image("../plots_R/p1.png")
        
    expander = st.beta_expander("Code-Monkey: exploratory plot")
    
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.write("Python code")
        with col2:
            st.write("R code")

    
elif choice == "Complete Pooling (model 1)":
    
    # python model (fun_models.py)
    py_model = '''
    with pm.Model() as m1:
        
        # shared variables
        t_shared = pm.Data('t_shared', t)
        
        # specify priors for parameters & model error
        beta = pm.Normal("beta", mu = 0, sigma = 1)
        alpha = pm.Normal("alpha", mu = 0, sigma = 1)
        sigma = pm.HalfNormal("sigma", sigma = 1)
        
        # calculate mu
        mu = alpha + beta * t_shared
        
        # likelihood
        y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y)
    '''
    
    # r model 
    r_model = '''
    # model specification
    f1 <- bf(y ~ t)
    
    # set priors
    prior_f1 <- c(
        prior(normal(0, 1), class = b),
        prior(normal(0, 1), class = Intercept),
        prior(normal(0, 1), class = sigma)
    )
    
    # fit model
    m1_prior <- brm(
        formula = f1,
        family = gaussian,
        data = train,
        prior = prior_f1,
        backend = "cmdstanr", # faster than rstan
        file = "
    )
    
    '''
    

    
    r'''
    ## Candidate model $1$ (Complete Pooling)
    Our first candidate model will be a complete pooling model.
    This is in contrast to (1) a no pooling model (i.e. all ID's 
    analyzed individually) and (2) a multilevel model
    (i.e. hierarchical random effects structure). 
    '''
    
    '''
    # Model specification
    '''
    
    st.latex(r''' 
        y_i \sim Normal(\mu, \sigma) \\
        \mu = \alpha + \beta \cdot x_i \\
        \alpha \sim Normal(0, 1) \\
        \beta \sim Normal(0, 1) \\
        \sigma \sim HalfNormal(0, 1)
        ''')
    
    '''
    # Compile model
    '''
    
    col1, col2 = st.beta_columns(2)
    
    
    '''
    # Prior predictive checks
    
    There are different levels of priors, see: https://jrnold.github.io/bayesian_notes/priors.html
    
    Our main model is run with what they refer to as a "generic weakly informative prior".
    
    Feel free to explore what happens with a much more informative prior, or with a very weak prior.
    
    NB: Notice the x-axis. 
    
    '''
    
    selection1 = st.radio(
        "Choose prior level for prior predictive", 
        ("Very Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_prior, col2_prior = st.beta_columns(2)
    
    translation_dct = {
        'Very Weak (sd = 5)': 'vague',
        'Generic (sd = 0.5)': 'reasonable',
        'Specific (sd = 0.05)': 'strict'}
    
    prior_level1 = translation_dct.get(selection1)

    with col1_prior: 
        st.image(f"../plots_python/pooled_{prior_level1}_prior_pred.jpeg")
        
    with col2_prior: 
        st.image(f"../plots_R/pooled_{prior_level1}_prior_pred.png")
    
    
    '''
    # Plate Notation 
    
    There are many reasons why a prior predictive check can be bad
    
    (including of course, bad priors). Something that is really nice in
    
    pyMC3 though is that you can check whether you actually specified the
    
    model as you intended to. Your model is shown in plate notation,
    
    which can seem confusing, and which I think is less intuitive than the
    
    really nice Kruschke diagrams/plots (link). However, it is still useful. 
    
    '''
    
    st.image("../plots_python/pooled_plate.png")
    
    '''
    # Posterior Predictive checks 
    '''
    
    selection2 = st.radio(
        "Choose prior level for posterior predictive", 
        ("Very Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_posterior, col2_posterior = st.beta_columns(2)
    
    prior_level2 = translation_dct.get(selection2)
    
    with col1_posterior:
        st.image(f"../plots_python/pooled_{prior_level2}_posterior_pred.jpeg")
        
    with col2_posterior:
        st.image(f"../plots_R/pooled_{prior_level2}_posterior_pred.png")
        
    
    '''
    # Check traces (sampling)
    
    When I am working in pyMC3 I like to check my traces at this point. 
    
    This is just one of the things that seems easier to me in pyMC3, 
    
    so I will only here show the pyMC3 code and output. 
    '''
    
    selection3 = st.radio(
        "Choose prior level for trace plot", 
        ("Very Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_level3 = translation_dct.get(selection3)
    
    st.image(f"../plots_python/pooled_{prior_level3}_plot_trace.jpeg")
    
    '''
    
    For all prior levels we see healthy traces to the right (catterpillars),
    
    and we see reasonably smooth and well-mixed KDE/histograms to the right. 
    
    However, notice that the values for our parameters differ based on our priors.
    
    The models fitted with either "Very Weak" or "Generic" priors are close, 
    
    but the "Specific" priors bias the model heavily towards our priors. 
    
    see: https://oriolabril.github.io/oriol_unraveled/python/arviz/matplotlib/2020/06/20/plot-trace.html
    
    for more details on Arviz' plot_trace() and customization.  
    '''
    
    '''
    # Summary
    
    We are now happy with our prior- and posterior predictive checks
    
    & with our sampling and chains. 
    
    We can now get an overview with the summary method. 
    '''
    
    selection4 = st.radio(
        "Choose prior level for summary", 
        ("Very Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_level4 = translation_dct.get(selection4)
    
    st.image(f"../plots_python/pooled_{prior_level4}_summary.png")

elif choice == "Multilevel (model 2)":
    
    r'''
    Multilevel model with: 
    
    1. Random intercepts ($\alpha$) &
    
    2. Random slopes ($\beta$)
    '''
    # put in the actual stuff here.
    st.latex(r''' 
    y_i \sim Normal(\mu, \sigma) \\
    \mu = \alpha + \beta \cdot x_i \\
    \alpha \sim Normal(0, 1) \\
    \beta \sim Normal(0, 1) \\
    \sigma \sim HalfNormal(0, 1)
    ''')
    
elif choice == "Student-t (model 3)":
    
    '''
    Multilevel model (as above) with Student-t likelihood function.
    
    Notice that only the top row is different. 
    '''
    # put in the actual stuff here.
    st.latex(r''' 
    y_i \sim Normal(\mu, \sigma) \\
    \mu = \alpha + \beta \cdot x_i \\
    \alpha \sim Normal(0, 1) \\
    \beta \sim Normal(0, 1) \\
    \sigma \sim HalfNormal(0, 1)
    ''')

elif choice == "Model Comparison":
    pass

elif choice == "Prediction": 
    pass

elif choice == "Sandbox":
    pass
