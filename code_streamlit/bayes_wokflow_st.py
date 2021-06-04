# import packages
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import arviz as az
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

# translation dictionaries: 
translation_prior = {
    'Weak (sd = 5)': 'weak',
    'Generic (sd = 0.5)': 'generic',
    'Specific (sd = 0.05)': 'specific'}

translation_uncertainty = {
    'population mean (fixed)': 'fixed',
    'individuals (full uncertainty)': 'full'
    }

translation_model = {
    'Pooled': 'pooled',
    'Multilevel': 'multilevel',
    'Student-t': 'student'}

# other layout options?
# 1. taking less space
# 2. perhaps not
choice = st.sidebar.radio('Sections:', ["Introduction", "Simulation & EDA", "Complete Pooling (model 1)", "Multilevel (model 2)", "Student-t (model 3)", "Model Comparison", "Prediction", "References & Inspiration"])


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
    
    ## shown code 
    
    You can copy the code chunks in the upper right corner
    
    if you want to follow along with the coding. 
    
    Note that the code I show in these sections is only for what I will
    
    call the "generic" prior level. You will have to change the priors
    
    (see github) to reproduce the "weak" and the "specific" prior levels. 
    
    ## R/brms vs python/pyMC3
    
    You will probably notice that some parts of the analysis require more
    
    lines of code in python and that some parts of the analysis require more
    
    lines of code in R. There are pros and cons to both languages 
    
    (say what they are). 
    
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
        st.image("../plots_python/EDA.jpeg")
        
    with col2: 
        st.image("../plots_R/EDA.png")
        
    expander = st.beta_expander("Code-Monkey: exploratory plot")
    
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.write("Python code")
        with col2:
            st.write("R code")

    
elif choice == "Complete Pooling (model 1)":
    
    model_context = "pooled"
    
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
    # Specify model
    '''
    
    ## python model
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
    
    ## r model    # r model 
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
    
    col1_code_model, col2_code_model = st.beta_columns(2)
    
    with col1_code_model:
        st.code(py_model)
    
    with col2_code_model: 
        st.code(r_model)
    
    '''
    # Prior predictive checks
    
    There are different levels of priors, see: https://jrnold.github.io/bayesian_notes/priors.html
    
    Our main model is run with what they refer to as a "generic weakly informative prior".
    
    Feel free to explore what happens with a much more informative prior, or with a very weak prior.
    
    NB: Notice the x-axis. 
    
    '''
    
    selection_pp_pool = st.radio(
        "Choose prior level for prior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp_pool, col2_pp_pool = st.beta_columns(2)
    
    prior_pp_pool = translation_prior.get(selection_pp_pool)

    with col1_pp_pool: 
        st.image(f"../plots_python/{model_context}_{prior_pp_pool}_prior_pred.jpeg")
        
    with col2_pp_pool: 
        st.image(f"../plots_R/{model_context}_{prior_pp_pool}_prior_pred.png")
    
    
    '''
    # Plate Notation 
    
    There are many reasons why a prior predictive check can be bad
    
    (including of course, bad priors). Something that is really nice in
    
    pyMC3 though is that you can check whether you actually specified the
    
    model as you intended to. Your model is shown in plate notation,
    
    which can seem confusing, and which I think is less intuitive than the
    
    really nice Kruschke diagrams/plots (link). However, it is still useful. 
    
    '''
    
    st.image(f"../plots_python/{model_context}_plate.png")
    
    '''
    # Check traces (sampling)
    '''
    
    selection_trace_pool = st.radio(
        "Choose prior level for trace plot", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_trace_pool, col2_trace_pool = st.beta_columns(2)
    
    prior_trace_pool = translation_prior.get(selection_trace_pool)
    
    with col1_trace_pool: 
        st.image(f"../plots_python/{model_context}_{prior_trace_pool}_plot_trace.jpeg")
    with col2_trace_pool: 
        st.image(f"../plots_R/{model_context}_{prior_trace_pool}_plot_trace.png")
    
    
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
    # Posterior Predictive checks 
    
    We have now accepted our prior predictive check and verified
    
    that computation (sampling) was okay. 
    
    We will now look at posterior predictive checks. 
    
    '''
    
    selection_pp2_pool = st.radio(
        "Choose prior level for posterior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp2_pool, col2_pp2_pool = st.beta_columns(2)
    
    prior_pp2_pool = translation_prior.get(selection_pp2_pool)
    
    with col1_pp2_pool:
        st.image(f"../plots_python/{model_context}_{prior_pp2_pool}_posterior_pred.jpeg")
        
    with col2_pp2_pool:
        st.image(f"../plots_R/{model_context}_{prior_pp2_pool}_posterior_pred.png")
        
    
    
    '''
    # Summary
    
    We are now happy with our prior- and posterior predictive checks
    
    & with our sampling and chains. 
    
    We can now get an overview with the summary method. 
    '''
    
    selection_summary = st.radio(
        "Choose prior level for summary", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_summary = translation_prior.get(selection_summary)
    
    st.image(f"../plots_python/{model_context}_{prior_summary}_summary.png")

    '''
    # HDI (vs. data)
    
    Something about how there are different things we can look at,
    
    e.g. fixed effects, all uncertainty (three levels actually). 
    
    '''
    
    col1_hdi, col2_hdi = st.beta_columns(2)
    
    with col1_hdi:
        selection_hdi1 = st.radio(
            "Choose prior level for HDI plots", 
            ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
            index = 1)
    
    with col2_hdi: 
        selection_hdi2 = st.radio(
            "See predictions for population mean (fixed effects) or for individuals (full uncertainty)",
            ("population mean (fixed)", "individuals (full uncertainty)"),
            index = 0
            )
    
    col1_hdiplot, col2_hdiplot = st.beta_columns(2)
    
    prior_level_hdi = translation_prior.get(selection_hdi1)
    hdi_type = translation_uncertainty.get(selection_hdi2)
    
    
    with col1_hdiplot:
        st.image(f"../plots_python/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.jpeg")
        
    with col2_hdiplot:
        st.image(f"../plots_R/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.png")
    
    '''
    # HDI (parameters)
    
    The last thing we might want to check is 
    
    how the model has estimated the parameters we care about. 
    
    '''
    
    selection_param = st.radio(
            "Choose prior level for HDI parameter plots", 
            ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
            index = 1)
    
    col1_hdi_param, col2_hdi_param = st.beta_columns(2)
    prior_param = translation_prior.get(selection_param)
    
    with col1_hdi_param: 
        st.image(f"../plots_python/{model_context}_{prior_param}_HDI_param.jpeg")
    
    with col2_hdi_param: 
        st.image(f"../plots_R/{model_context}_{prior_param}_HDI_param.png")

elif choice == "Multilevel (model 2)":
    
    model_context = "multilevel"
    
    r'''
    Multilevel model with: 
    
    1. Random intercepts ($\alpha$) &
    
    2. Random slopes ($\beta$)
    
    NB: not taking into account (in the pyMC3 model) the correlation 
    
    between intercepts and slopes (which is the "lkj(1)" parameter in brms).
    
    For more, see: https://docs.pymc.io/notebooks/multilevel_modeling.html
    '''
    # put in the actual stuff here.
    st.latex(r''' 
    y_i \sim Normal(\mu, \sigma) \\
    \mu = \alpha + \beta \cdot x_i \\
    \alpha \sim Normal(0, 1) \\
    \beta \sim Normal(0, 1) \\
    \sigma \sim HalfNormal(0, 1)
    ''')
    
    '''
    # Prior predictive checks
    
    There are different levels of priors, see: https://jrnold.github.io/bayesian_notes/priors.html
    
    Our main model is run with what they refer to as a "generic weakly informative prior".
    
    Feel free to explore what happens with a much more informative prior, or with a very weak prior.
    
    NB: Notice the x-axis. 
    
    '''
    
    selection_pp = st.radio(
        "Choose prior level for prior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp, col2_pp = st.beta_columns(2)
    
    prior_pp = translation_prior.get(selection_pp)

    with col1_pp: 
        st.image(f"../plots_python/{model_context}_{prior_pp}_prior_pred.jpeg")
        
    with col2_pp: 
        st.image(f"../plots_R/{model_context}_{prior_pp}_prior_pred.png")
    
    
    '''
    # Plate Notation 
    
    There are many reasons why a prior predictive check can be bad
    
    (including of course, bad priors). Something that is really nice in
    
    pyMC3 though is that you can check whether you actually specified the
    
    model as you intended to. Your model is shown in plate notation,
    
    which can seem confusing, and which I think is less intuitive than the
    
    really nice Kruschke diagrams/plots (link). However, it is still useful. 
    
    '''
    
    st.image(f"../plots_python/{model_context}_plate.png")
    
    
    '''
    # Check traces (sampling)
    
    '''
    
    selection_trace = st.radio(
        "Choose prior level for trace plot", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_trace, col2_trace = st.beta_columns(2)
    
    prior_trace = translation_prior.get(selection_trace)
    
    with col1_trace: 
        st.image(f"../plots_python/{model_context}_{prior_trace}_plot_trace.jpeg")
    with col2_trace: 
        st.image(f"../plots_R/{model_context}_{prior_trace}_plot_trace.png")
    
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
    # Posterior Predictive checks 
    
    We have now accepted our prior predictive check and verified
    
    that computation (sampling) was okay. 
    
    We will now look at posterior predictive checks. 
    
    '''
    
    selection_pp2 = st.radio(
        "Choose prior level for posterior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp2, col2_pp2 = st.beta_columns(2)
    
    prior_pp2 = translation_prior.get(selection_pp2)
    
    with col1_pp2:
        st.image(f"../plots_python/{model_context}_{prior_pp2}_posterior_pred.jpeg")
        
    with col2_pp2:
        st.image(f"../plots_R/{model_context}_{prior_pp2}_posterior_pred.png")
        
    '''
    # Summary
    
    We are now happy with our prior- and posterior predictive checks
    
    & with our sampling and chains. 
    
    We can now get an overview with the summary method. 
    '''
    
    selection_summary = st.radio(
        "Choose prior level for summary", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_summary = translation_prior.get(selection_summary)
    
    st.image(f"../plots_python/{model_context}_{prior_summary}_summary.png")

    '''
    # HDI (vs. data)
    
    Something about how there are different things we can look at,
    
    e.g. fixed effects, all uncertainty (three levels actually). 
    
    '''
    
    col1_hdi, col2_hdi = st.beta_columns(2)
    
    with col1_hdi:
        selection_hdi1 = st.radio(
            "Choose prior level for HDI plots", 
            ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
            index = 1)
    
    with col2_hdi: 
        selection_hdi2 = st.radio(
            "See predictions for population mean (fixed effects) or for individuals (full uncertainty)",
            ("population mean (fixed)", "individuals (full uncertainty)"),
            index = 0
            )
    
    col1_hdiplot, col2_hdiplot = st.beta_columns(2)

    prior_level_hdi = translation_prior.get(selection_hdi1)
    hdi_type = translation_uncertainty.get(selection_hdi2)
    
    
    with col1_hdiplot:
        st.image(f"../plots_python/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.jpeg")
        
    with col2_hdiplot:
        st.image(f"../plots_R/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.png")
    
    '''
    # HDI (parameters)
    
    The last thing we might want to check is 
    
    how the model has estimated the parameters we care about. 
    
    '''
    
    selection_param = st.radio(
            "Choose prior level for HDI parameter plots", 
            ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
            index = 1)
    
    col1_hdi_param, col2_hdi_param = st.beta_columns(2)
    prior_param = translation_prior.get(selection_param)
    
    with col1_hdi_param: 
        st.image(f"../plots_python/{model_context}_{prior_param}_HDI_param.jpeg")
    
    with col2_hdi_param: 
        st.image(f"../plots_R/{model_context}_{prior_param}_HDI_param.png") 
    
elif choice == "Student-t (model 3)":
    
    model_context = "student"
    
    r'''
    Still multilevel (random slopes and intercepts).
    
    Notice that only top row has changed. 
    '''
    # put in the actual stuff here.
    st.latex(r''' 
    y_i \sim Normal(\mu, \sigma) \\
    \mu = \alpha + \beta \cdot x_i \\
    \alpha \sim Normal(0, 1) \\
    \beta \sim Normal(0, 1) \\
    \sigma \sim HalfNormal(0, 1)
    ''')
    
    '''
    # Prior predictive checks
    
    There are different levels of priors, see: https://jrnold.github.io/bayesian_notes/priors.html
    
    Our main model is run with what they refer to as a "generic weakly informative prior".
    
    Feel free to explore what happens with a much more informative prior, or with a very weak prior.
    
    NB: Notice the x-axis. 
    
    '''
    
    selection_pp = st.radio(
        "Choose prior level for prior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp, col2_pp = st.beta_columns(2)
    
    prior_pp = translation_prior.get(selection_pp)

    with col1_pp: 
        st.image(f"../plots_python/{model_context}_{prior_pp}_prior_pred.jpeg")
        
    with col2_pp: 
        st.image(f"../plots_R/{model_context}_{prior_pp}_prior_pred.png")
    
    
    '''
    # Plate Notation 
    
    There are many reasons why a prior predictive check can be bad
    
    (including of course, bad priors). Something that is really nice in
    
    pyMC3 though is that you can check whether you actually specified the
    
    model as you intended to. Your model is shown in plate notation,
    
    which can seem confusing, and which I think is less intuitive than the
    
    really nice Kruschke diagrams/plots (link). However, it is still useful. 
    
    '''
    
    st.image(f"../plots_python/{model_context}_plate.png")
    
    
    '''
    # Check traces (sampling)
    
    '''
    
    selection_trace = st.radio(
        "Choose prior level for trace plot", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_trace, col2_trace = st.beta_columns(2)
    
    prior_trace = translation_prior.get(selection_trace)
    
    with col1_trace: 
        st.image(f"../plots_python/{model_context}_{prior_trace}_plot_trace.jpeg")
    with col2_trace: 
        st.image(f"../plots_R/{model_context}_{prior_trace}_plot_trace.png")
    
    
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
    # Posterior Predictive checks 
    
    We have now accepted our prior predictive check and verified
    
    that computation (sampling) was okay. 
    
    We will now look at posterior predictive checks. 
    
    '''
    
    selection_pp2 = st.radio(
        "Choose prior level for posterior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp2, col2_pp2 = st.beta_columns(2)
    
    prior_pp2 = translation_prior.get(selection_pp2)
    
    with col1_pp2:
        st.image(f"../plots_python/{model_context}_{prior_pp2}_posterior_pred.jpeg")
        
    with col2_pp2:
        st.image(f"../plots_R/{model_context}_{prior_pp2}_posterior_pred.png")
        
    
    
    '''
    # Summary
    
    We are now happy with our prior- and posterior predictive checks
    
    & with our sampling and chains. 
    
    We can now get an overview with the summary method. 
    '''
    
    selection_summary = st.radio(
        "Choose prior level for summary", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_summary = translation_prior.get(selection_summary)
    
    st.image(f"../plots_python/{model_context}_{prior_summary}_summary.png")

    '''
    # HDI (vs. data)
    
    Something about how there are different things we can look at,
    
    e.g. fixed effects, all uncertainty (three levels actually). 
    
    '''
    
    col1_hdi, col2_hdi = st.beta_columns(2)
    
    with col1_hdi:
        selection_hdi1 = st.radio(
            "Choose prior level for HDI plots", 
            ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
            index = 1)
    
    with col2_hdi: 
        selection_hdi2 = st.radio(
            "See predictions for population mean (fixed effects) or for individuals (full uncertainty)",
            ("population mean (fixed)", "individuals (full uncertainty)"),
            index = 0
            )
    
    col1_hdiplot, col2_hdiplot = st.beta_columns(2)
    
    prior_level_hdi = translation_prior.get(selection_hdi1)
    hdi_type = translation_uncertainty.get(selection_hdi2)
    
    with col1_hdiplot:
        st.image(f"../plots_python/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.jpeg")
        
    with col2_hdiplot:
        st.image(f"../plots_R/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.png")
    
    '''
    # HDI (parameters)
    
    The last thing we might want to check is 
    
    how the model has estimated the parameters we care about. 
    
    '''
    
    selection_param = st.radio(
            "Choose prior level for HDI parameter plots", 
            ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
            index = 1)
    
    col1_hdi_param, col2_hdi_param = st.beta_columns(2)
    prior_param = translation_prior.get(selection_param)
    
    with col1_hdi_param: 
        st.image(f"../plots_python/{model_context}_{prior_param}_HDI_param.jpeg")
    
    with col2_hdi_param: 
        st.image(f"../plots_R/{model_context}_{prior_param}_HDI_param.png")

elif choice == "Model Comparison":
    
    prior_context = "generic"
    
    '''
    # Model comparison
    
    We will only now compare the models with what I have
    
    called "generic" priors (sigma, sd = 0.5). 
    
    I like to compare models in two ways: 
    
    1. Using information criteria (loo). 
    
    2. Using posterior predictive checks (& potentially predictions on unseen data). 
    
    If you want to know more check the 'Concept-Guru' section. 
    
    NB: WE NEED BOTH PREDICTIVE AND FIT!! (different uncertainty)...
    (CH4 in recoded for this difference) (CH 12 in recoded for multilevel).
    Depends on whether we want to plot plausible average height or 
    whether we want to plot simulated heights of individuals. 
    Obviously there will be more variability in the second one 
    (see recoded CH4 and McElreath). 
    
    '''
    
    expander = st.beta_expander("Concept-Guru: Model Comparison")
    
    with expander: 
        
        '''
        The second one always works. Since Bayesian models are always generative (see McElreath)
        
        we can simulate/generate new data based on our fitted distributions over parameters
        
        and our likelihood function. 
        
        The first is typically invalid if we want to compare models with different
        
        likelihood functions. In our specific case however (Gaussian and Student-t) however,
        
        it is valid (find source). 
        '''
    
    
    '''
    # Compare posterior predictive
    
    Which one do you think looks the most reasonable?
    
    Are there any of the models that look particularly worse?
    
    '''
    
    selection_pp = st.radio(
        "Choose model type to display posterior predictive checks for ", 
        ("Pooled", "Multilevel", "Student-t"),
        index = 1)
    
    col1_pp, col2_pp = st.beta_columns(2)
    
    model_pp = translation_model.get(selection_pp)

    with col1_pp: 
        st.image(f"../plots_python/{model_pp}_{prior_context}_posterior_pred.jpeg")
        
    with col2_pp: 
        st.image(f"../plots_R/{model_pp}_{prior_context}_posterior_pred.png")
    
    '''
    # Compare HDI
    
    Now lets look at model predictions,
    
    both for fixed effects only, and with the full uncertainty. 
    
    Again: which model do you think we should prefer?
    
    '''
    
    col1_hdi1, col2_hdi1 = st.beta_columns(2)
    
    with col1_hdi1:
        selection_hdi1 = st.radio(
            "Choose model type to display HDI prediction intervals for", 
            ("Pooled", "Multilevel", "Student-t"),
            index = 1)
    
    with col2_hdi1: 
        selection_hdi2 = st.radio(
            "See predictions for population mean (fixed effects) or for individuals (full uncertainty)",
            ("population mean (fixed)", "individuals (full uncertainty)"),
            index = 0
            )
    
    model_hdi = translation_model.get(selection_hdi1)
    uncertainty = translation_uncertainty.get(selection_hdi2)

    col1_hdi2, col2_hdi2 = st.beta_columns(2)
    
    with col1_hdi2: 
        st.image(f"../plots_python/{model_hdi}_{prior_context}_HDI_{uncertainty}.jpeg")
        
    with col2_hdi2: 
        st.image(f"../plots_R/{model_hdi}_{prior_context}_HDI_{uncertainty}.png")
    
    '''
    # Information criterion (loo)
    
    We have now done some eye-balling and will now check what
    
    loo has to say. 
    
    '''
    
    st.image(f"../plots_python/loo_comparison.png")
    
    '''
    Overthinking, that loo is leave-one-out approximation. 
    
    Here it is indicated that pooled model "underfits" and that
    
    student-t model "overfits" (i.e. the extra parameter (nu "v"))
    
    is not giving us enough to earn its keep. 
    
    '''

elif choice == "Prediction": 
    
    '''
    # Prediction on unseen data
    
    This will be only a brief introduction to prediction in pyMC3/brms,
    
    and will only cover predicting (1) groups that are already in the data
    
    and (2) only the trend for the group (not clusters / individuals). 
    
    If you want to see more prediction (e.g. on new groups or on individuals)
    
    Then please let me know. The process is largely the same though, and 
    
    you can check the "References & Inspiration" page to see where to go next. 
    
    In this section we will only be predicting based on the "multilevel model". 
    
    '''
    
    '''
    
    # HDI prediction interval 
    
    (perhaps we should at least have both mean forecast and individual forecast).
    
    '''
    
    col1, col2 = st.beta_columns(2)
    
    with col1: 
        st.image("../plots_python/multilevel_generic_HDI_predictions.jpeg") 
    
    with col2: 
        st.image("../plots_R/multilevel_generic_HDI_predictions.png")
    
elif choice == "References & Inspiration":
    pass
