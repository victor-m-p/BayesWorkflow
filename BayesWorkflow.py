# import packages
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import arviz as az
import CodeText as ct

# we want this for all pages
st.set_page_config(
    page_title="Bayesian Workflow",
    page_icon="bayes_bois.png",
    layout="wide",
    initial_sidebar_state="expanded",
    )

st.image("img/bayes_bois.png", width = 100)

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
    'Intercept': 'intercept',
    'Covariation': 'covariation'}

translation_mod_name = {
    'Pooled': 'm_pooled',
    'Intercept': 'm_intercept',
    'Covariation': 'm_covariation'}

translation_idata = {
    'Pooled': 'pooled_idata',
    'Intercept': 'intercept_idata',
    'Covariation': 'covariation_idata'
}

translation_code = {
    'population mean (fixed)': ('add_fitted_draws', '.value'),
    'individuals (full uncertainty)': ('add_predicted_draws', '.prediction')
}

translation_sigma = {
    'Weak (sd = 5)': '5',
    'Generic (sd = 0.5)': '0.5',
    'Specific (sd = 0.05)': '0.05'
}

# other layout options?
# 1. taking less space
# 2. perhaps not
choice = st.sidebar.radio('Sections:', ["Introduction", "Simulation & EDA", "Complete Pooling (model 1)", "Random Intercepts (model 2)", "Multilevel Covariation (model 3)", "Model Comparison", "Prediction", "References & Inspiration"])


if choice == "Introduction":
    
    '''
    # Get in touch

    [![Github Follow](https://img.shields.io/github/followers/victor-m-p?style=social)](https://github.com/victor-m-p/BayesWorkflow) &nbsp; [![Twitter Follow](https://img.shields.io/twitter/follow/vic_moeller?style=social)](https://twitter.com/vic_moeller)
    
    Please get in touch if you find the app interesting or have suggestions for improvements, have hit upon bugs or would like to see 
    something covered which is not implemented at present. The app is a work in progress (and will probably stay that way) so I 
    am very interested in feedback which can help me to improve this ressource.
    
    '''
    
    
    '''
    # Purpose 
    
    This notebook attempts to show how to carry out two (almost) identical bayesian analyses, using **pyMC3** (python) and **brms** (R). 
    The main purpose of this app is to show users with experience in one language how to carry out a bayesian analysis in the other. 
    I have specifically had people who are looking to transition from brms to pyMC3 in mind, but people coming from python should also find the app interesting.
    I have tried to explain the code and concepts in selected parts, but this is not supposed to be an exhaustive guide to bayesian statistics nor pyMC3 or brms. 
    If you want to dive deeper into either bayesian statistics or more advanced analyses I have provided a list of good resources in the appendix (References & Inspiration).
    
    '''
    
    '''
    # How to use 
    
    For all parts of the analysis (for both languages) you can access reproducible code, and easily copy it to clipboard.
    I hope that this will encourage you to run the code alongside the app, since this is the only way to really understand what is going on.
    While building a bridge between pyMC3 and brms is the main objective, I hope that parts of the analysis & workflow might also lead you to a better bayesin workflow.
    
    '''
    
    '''
    # Functionality 
    
    You will see boxes with the titles **Code-Monkey**, **Language-Learner** and **Concept-Guru**. These let you dive deeper into the material:
    
    * :monkey: *Code-Monkey*: Display code to reproduce analysis
    * :keyboard: *Language-Learner*: Explanations of code-implementation differences between python/pyMC3 & R/brms
    * :male_mage: *Concept-Guru*: Conceptual deep-dives

    '''
    
    '''
    # Bayesian Workflow
    '''
    col1, col2 = st.beta_columns(2)
    
    with col1: 
            '''
        The notebook follows a workflow close to what is presented in Gelman 2020 (see figure). 
        Some parts are not included (yet), but we will cover the following: 
        
        1. Fake data simulation 
        
        2. Picking an initial model 
        
        3. Prior predictive checks
        
        4. Fitting a model 
        
        5. Validate computation
        
        6. Posterior predictive checks
        
        7. Model Comparison
        
        8. Prediction on unseen data
        
        '''
    with col2: 
        st.image("img/Gelman.png")


    
    
elif choice == "Simulation & EDA":

    '''
    # Data (Simulation)

    For this analysis we will simulate our own data. This is nice, because we will know the *true* parameter values. 
    We can spin our own story about what the data corresponds to. 
    
    * ```x-value``` corresponds to consecutive years (t) 
    
    * ```y-value``` corresponds to grading on the danish citizenship test
    
    * ```ID-value``` corresponds to individual aliens
    
    Based on 15 consecutive years of data (t) and corresponding gradings (y) and a sample of 15 aliens (ID) we want to infer how fast how aliens learn about danish culture (beta), and how much they know when they arrive (alpha). 
    We might also be interested in the variability between aliens or our left-over uncertainty, but let's table that for now. 
    Remember to check out the :monkey: *Code-Monkey* boxes to follow along with the code. 
    
    '''
    
    ### code ###
    expander = st.beta_expander("🐒 Code-Monkey: Reproducibility")
    py_reproducibility = ct.py_reproducibility()
    
    with expander: 
        st.code(py_reproducibility)
        
    
    ### code ##
    expander = st.beta_expander('🐒 Code-Monkey: Simulation')
    py_sim = ct.py_sim()
    
    with expander: 
        st.code(py_sim)
            
    '''
    # Quick EDA
    
    Below is a scatter plot of the *training* data, where a regression line is showed for each alien (ID). 
    
    '''
    
    ### plot ###
    col1_EDA, col2_EDA = st.beta_columns(2)
    
    with col1_EDA: 
        st.image("plots_python/EDA.jpeg")
    with col2_EDA: 
        st.image("plots_R/EDA.png")
    
    ### code ###
    expander = st.beta_expander("🐒 Code-Monkey: EDA")
    
    py_EDA = ct.py_EDA()
    R_EDA = ct.R_EDA()
    
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1:
            st.code(py_EDA)
        with col2: 
            st.code(R_EDA)

    
elif choice == "Complete Pooling (model 1)":
    
    # for f-strings.
    model_context = "pooled"
    model_name = "m_pooled" # should be in args
    model_formula = "f_pooled" # should be in args
    model_family = "gaussian" 
    prior_name = "prior_pooled" # should be in args
    data_type = "train"
    idata_name = "pooled_idata"
    
    '''
    # Candidate model 1 (Complete Pooling)
    
    Our first candidate model will be a complete pooling model. 
    This model treats each observation at each time-point as if it belongs to the same alien (ID). 
    You might already feel that this is not a satisfactory model, but bear with me for now.
    We will build on top of what we have learned and get to more complex models in the next sections. 
    
    '''
    
    ### code ###
    py_reproducibility = ct.py_reproducibility()
    R_reproducibility = ct.R_reproducibility()
    
    expander = st.beta_expander("🐒 Code-Monkey: Packages & Reproducibility")
    
    with expander: 
        col1_reproducibility, col2_reproducibility = st.beta_columns(2)
        with col1_reproducibility:
            st.code(py_reproducibility)
        with col2_reproducibility: 
            st.code(R_reproducibility) 
    
    ### code ###
    py_preprocessing = ct.py_preprocessing()
    R_preprocessing = ct.R_preprocessing()
    
    expander = st.beta_expander("🐒 Code-Monkey: Preprocessing")
    
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1:
            st.code(py_preprocessing)
        with col2: 
            st.code(R_preprocessing) 
    
    '''
    # Model specification (math)
    
    We can formulate the complete pooling model (with generic priors)
    
    '''
    
    st.latex(r''' 
        y_i \sim Normal(\mu_i, \sigma) \\
        \mu = \alpha + \beta \cdot x_i \\
        \alpha \sim Normal(1.5, 0.5) \\
        \beta \sim Normal(0, 0.5) \\
        \sigma \sim HalfNormal(0.5)
        ''')
    
    '''
    # Model specification (code)
    
    Now we need to translate this into pyMC3 and brms code. 
    Throughout the app you can choose which prior-level to display code & plots for. 
    I have run all analysis for three levels of priors, a generic prior (sigma and sd = 0.5)
    is shown by default, a specific prior (sigma and sd = 0.05) and a weak prior (sigma and sd = 5).
    For more on priors, see [Gelman's prior choice recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)
    
    '''
    
    ### code prep ###
    selection_prior = st.radio(
        "Choose prior level (how uncertain)", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_choice = translation_sigma.get(selection_prior)
    
    ### code ###
    py_model = ct.py_pooled(model_name, prior_choice)
    r_model = ct.R_pooled(model_name, model_formula, prior_name, prior_choice)

    expander = st.beta_expander("🐒 Code-Monkey: Model specification")
    with expander: 
        col1_model, col2_model = st.beta_columns(2)
        
        with col1_model:
            st.code(py_model)
        
        with col2_model: 
            st.code(r_model)
    
    ### :keyboard: Language-Learner ###
    expander = st.beta_expander("⌨️ Language-Learner: Model specification")
    
    with expander: 

            '''
            
            # model formula vs. manual specification: 
            
            In brms we specify our model via a model formula (see right). In pyMC3 we can either explicitly specify our model,
            or we can do it via a model formula (implemented in the glm module). I will use the explicit notation here, because it 
            forces me to understand how the model works. The model formula might seem more intuitive or user friendly, 
            but I think that it is too easy to go wrong when we do not know what is happening under the hood. 
            
            for example, notice that we have specified the intercept with the special 0 + Intercept syntax, rather than the more common 1 + ... syntax. 
            We should do this if we have not mean-centered our predictors and are creating a model that is not an intercept only model (which is the case here).
            You can read more about this here: https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/horoscopes-insights.html#use-the-0-intercept-syntax
            
            # Shared Variables: 
            
            In the pyMC3 code (left) we have created something called a shared variable for our x variable, time (t). 
            This has to be done if we want to generate predictions based on new data (data that the model was not trained on).
            We only need to create a shared variable for t here, because it is the only predictor variable in the model. 
            pyMC3 relies on theano as the backend (just as brms relies on stan), and theano needs to encode variables that we might want to change in a different format. 
            
            # Sigma (Normal or HalfNormal?): 
            
            We have specified a normal distribution for the sigma distribution in the brms priors (right), but we have specified a half-normal distribution for the sigma distribution i pyMC3 (left). 
            This might seem confusing, but the prior for the sigma distribution cannot actually be a normal distribution (as this would allow negative values). 
            The reason why we can do this in brms is because it realizes that draws from the sigma distribution have to be positive. 
            So, when it encounters a negative value it throws away that sample. This will then effectively give us a half-normal distribution. 
            pyMC3 does not baby-sit in this way, and will throw an error if we try to specify a normal distribution for the sigma parameter. 
            You can try this for yourself. 
            '''
    
    '''
    # Plate Notation 
    
    Something that is really nice in pyMC3 is that we can check whether we specified the model as we intended to. 
    Our model is shown in plate notation, and which I think is less intuitive than the awesome [Kruschke diagrams](http://www.sumsar.net/blog/2013/10/diy-kruschke-style-diagrams/).
    They basically communicate the same though, and once we learn to read them they are a great sanity check.
    
    '''
    
    ### plot ###
    st.image(f"plots_python/{model_context}_plate.png")
    
    ### code ###
    py_plate = ct.py_plate(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Plate Notation")
    
    with expander: 
        st.code(py_plate)
    
    
    '''
    # Prior predictive checks
    
    The next thing we will want to assess is whether our priors are on a reasonable scale
    as compared to the data. We can get nice *prior predictive checks* in both pyMC3 and brms. 
    Here we plot 100 draws from our prior predictive (blue lines) against the true distribution of the data (black line).
    In pyMC3 a blue dotted line shows the mean draws from our prior. 
    
    '''
    
    ### plot ###
    selection_pp_pool = st.radio(
        "Choose prior level for prior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp_pool, col2_pp_pool = st.beta_columns(2)
    
    prior_pp_pool = translation_prior.get(selection_pp_pool)

    with col1_pp_pool: 
        st.image(f"plots_python/{model_context}_{prior_pp_pool}_prior_pred.jpeg")
        
    with col2_pp_pool: 
        st.image(f"plots_R/{model_context}_{prior_pp_pool}_prior_pred.png")
    
    ### code ###
    py_pp = ct.py_pp(model_name, idata_name)
    R_pp = ct.R_pp(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Prior predictive checks")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_pp)
        with col2:
            st.code(R_pp) 
            
    '''
    # Sample posterior
    
    We have now verified that the we have specified our model correctly (plate) and let's say that we are happy with our prior predictive checks.
    We should now sample the posterior. 
    
    '''
    
    ### code ###
    py_sample = ct.py_sample(model_name, idata_name)
    R_sample = ct.R_sample(model_name, model_formula, model_family, prior_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Sample posterior")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_sample)
        with col2:
            st.code(R_sample) 
    
    
    '''
    # Check traces (sampling)
    
    The first thing we might want to check now is wheather the sampling/computation was successfull. 
    I like to generate *trace plots* at this point. 
    If we observe issues at this point (e.g. divergences, poor mixing) 
    there are good in diagnostic tools available for [pyMC3 models](https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html)
    and for [brms models](https://mc-stan.org/bayesplot/).
    
    '''
    
    ### plot ###
    selection_trace_pool = st.radio(
        "Choose prior level for trace plot", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_trace_pool, col2_trace_pool = st.beta_columns(2)
    
    prior_trace_pool = translation_prior.get(selection_trace_pool)
    
    with col1_trace_pool: 
        st.image(f"plots_python/{model_context}_{prior_trace_pool}_plot_trace.jpeg")
    with col2_trace_pool: 
        st.image(f"plots_R/{model_context}_{prior_trace_pool}_plot_trace.png")
    
    ### code ###
    py_trace = ct.py_trace(idata_name)
    R_trace = ct.R_trace(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Trace-plot")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_trace)
        with col2:
            st.code(R_trace) 
    
    '''
    
    For all prior levels we see healthy traces to the right (catterpillars),
    and we see reasonably smooth and well-mixed KDE/histograms to the right. 
    However, notice that the values for our parameters differ based on our priors.
    The models fitted with either *weak* or *generic* priors have estimated the
    parameters similarly, 
    but the model fitted with *specific* priors restrict the posterior
    (i.e. the priors constrain the posterior unduly). 
    '''
    
    '''
    # Summary
    
    We can now (optionally) check the summary of the model. We might not be interested in the estimated parameters (yet)
    but the summary also gives us information about the number of **effective samples** and **R-hat values**. 
    
    '''
    
    ### plot ###
    selection_summary = st.radio(
        "Choose prior level for summary", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_summary = translation_prior.get(selection_summary)
    
    st.image(f"plots_python/{model_context}_{prior_summary}_summary.png")
    
    ### code ###
    py_summary = ct.py_summary(idata_name)
    R_summary = ct.R_summary(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Summary")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_summary)
        with col2:
            st.code(R_summary)
    
    '''
    # Posterior Predictive checks 
    
    If all is well so far we should now generate *posterior predictive checks*.  
    Similarly to *prior predictive checks* we have blue draws and a black line
    showing the true distribution. Check the *specific* prior and notice that 
    the posterior has not adapted well here (too constrained by the strong prior
    as discussed earlier).
    
    '''
    
    ### plot ### 
    selection_pp2_pool = st.radio(
        "Choose prior level for posterior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp2_pool, col2_pp2_pool = st.beta_columns(2)
    
    prior_pp2_pool = translation_prior.get(selection_pp2_pool)
    
    with col1_pp2_pool:
        st.image(f"plots_python/{model_context}_{prior_pp2_pool}_posterior_pred.jpeg")
        
    with col2_pp2_pool:
        st.image(f"plots_R/{model_context}_{prior_pp2_pool}_posterior_pred.png")

    ### code ###
    py_pp2 = ct.py_post_pred(model_name, idata_name)
    R_pp2 = ct.R_post_pred(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Posterior predictive")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_pp2)
        with col2:
            st.code(R_pp2) 
    
    
    ### Quiz ###
    expander = st.beta_expander("🧙‍♂️ Concept-Guru: Posterior Predictive Checks")
    #selection_quiz = st.multiselect(
    #    "QUIZ: The posterior predictive check indicates our model:", 
    #    ("reasonably captures patterns in the data (accept model)", "does not reliably capture patterns in the data (reject model)"),
    #    index = 1)
    with expander: 
        
        '''
        **QUIZ**: Based on the *posterior predictive check*, which do you think is an appropriate response? (choose one)
        '''
        
        option_a = st.checkbox('The model reliably captures the important patterns in the data (accept model)')
        option_b =  st.checkbox('The model does not reliably capture the important patterns in the data (reject model)')
        
        if option_a: 
        
            '''
            
            I disagree: We see that the mode of the true posterior distribution and the mode of our predictive draws differ systematically. 
            The posterior has long tails (is not normally distributed) which is not well captured by our model. 
            We should reject this model, and consider what we have missed. I often find that the issue is that the likelihood-function
            is improper, or that the model has not been specified with the appropriate random effects structure. 
            
            '''
        
        if option_b: 
            '''
            I agree: We see that the mode of the true posterior distribution and the mode of our predictive draws differ systematically. 
            The posterior has long tails (is not normally distributed) which is not well captured by our model. 
            We should reject this model, and consider what we have missed. I often find that the issue is that the likelihood-function
            is improper, or that the model has not been specified with the appropriate random effects structure. 
            '''
    
    '''
    # HDI (vs. data)
    
    We can also now run the model forward (i.e. generate predictions from the model). 
    We can compare these model predictions with the data that the model is trained on, to check whether the model has captured the patterns in the data.
    We can do this either for (a) fixed effects only or (b) with the full model uncertainty. 
    If we generate predictions for fixed effects only, we will get predictions for the *mean* of the population. 
    If we generate predictions with the full model uncertainty  (incl. sigma) we will get predictions for individuals.
    '''
    
    ### plot ###
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
        st.image(f"plots_python/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.jpeg")
        
    with col2_hdiplot:
        st.image(f"plots_R/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.png")
    
    ### code ###
    R_function, R_type = translation_code.get(selection_hdi2)
    R_hdi = ct.R_hdi_data_pool(model_name, R_type, data_type, R_function, hdi_type)
    
    if hdi_type == "fixed": 
        py_hdi = ct.py_hdi_data_fixed(hdi_type, idata_name)
    elif hdi_type == "full": 
        py_hdi = ct.py_hdi_data_full(hdi_type, idata_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: HDI prediction intervals")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_hdi)
        with col2:
            st.code(R_hdi) 
            
    '''
    # HDI (parameters)
    
    The last thing we might want to inspect at this point is the estimated distributions (and HDI intervals) for our inferred parameters. 
    
    '''
    
    ### plot ###
    selection_param = st.radio(
            "Choose prior level for HDI parameter plots", 
            ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
            index = 1)
    
    col1_hdi_param, col2_hdi_param = st.beta_columns(2)
    prior_param = translation_prior.get(selection_param)
    
    with col1_hdi_param: 
        st.image(f"plots_python/{model_context}_{prior_param}_HDI_param.jpeg")
    
    with col2_hdi_param: 
        st.image(f"plots_R/{model_context}_{prior_param}_HDI_param.png")

    ### code ###
    py_hdi_param = ct.py_hdi_param(idata_name)
    R_hdi_param = ct.R_hdi_param(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: HDI parameter intervals")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_hdi_param)
        with col2:
            st.code(R_hdi_param) 
    
elif choice == "Random Intercepts (model 2)":
    
    # for f-strings.
    model_context = "intercept"
    model_name = "m_intercept" # should be in args
    model_formula = "f_intercept" # should be in args
    model_family = "gaussian" 
    prior_name = "prior_intercept" # should be in args
    data_type = "train"
    idata_name = "intercept_idata"
    
    r'''
    # Candidate model 2 (Random intercepts)
    Our second candidate model will extend the first to include random intercepts ($\alpha$),
    and thus be our first multilevel (hierarchical) model. 
    
    '''
    
    ### code ###
    py_reproducibility = ct.py_reproducibility()
    R_reproducibility = ct.R_reproducibility()
    
    expander = st.beta_expander("🐒 Code-Monkey: Packages & Reproducibility")
    
    with expander: 
        col1_reproducibility, col2_reproducibility = st.beta_columns(2)
        with col1_reproducibility:
            st.code(py_reproducibility)
        with col2_reproducibility: 
            st.code(R_reproducibility) 
    
    ### code ###
    py_preprocessing = ct.py_preprocessing()
    R_preprocessing = ct.R_preprocessing()
    
    expander = st.beta_expander("🐒 Code-Monkey: Preprocessing")
    
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1:
            st.code(py_preprocessing)
        with col2: 
            st.code(R_preprocessing) 
    
    '''
    # Model specification (math) 
    
    Specification is for the pyMC3 model without parameter for the 
    correlation between random effects (lkj). Also here concretely
    specified for the generic prior. 
    
    '''
    
    st.latex(r''' 
        y_{i, j} \sim Normal(\mu_{i, j}, \sigma) \\
        \mu_{i, j} = \alpha_{var_j} + \beta \cdot x_i \\
        \alpha_{var_j} \sim Normal(\alpha_j, \alpha_{sigma_j}) \\
        \alpha \sim Normal(1.5, 0.5) \\
        \beta \sim Normal(0, 0.5) \\
        \alpha_{sigma} \sim HalfNormal(0, 0.5) \\
        \sigma \sim HalfNormal(0.5)
        ''')
    
    '''
    # Model specification (code)
    '''
    
    ### code prep ###
    selection_prior = st.radio(
        "Choose prior level (how uncertain)", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_choice = translation_sigma.get(selection_prior)
    
    ### code ###
    py_model = ct.py_intercept(model_name, prior_choice)
    r_model = ct.R_intercept(model_name, model_formula, prior_name, prior_choice)

    expander = st.beta_expander("🐒 Code-Monkey: Model specification")
    with expander: 
        col1_model, col2_model = st.beta_columns(2)
        
        with col1_model:
            st.code(py_model)
        
        with col2_model: 
            st.code(r_model)
    
    '''
    # Plate Notation 
    
    '''
    
    ### plot ###
    st.image(f"plots_python/{model_context}_plate.png")
    
    ### code ###
    py_plate = ct.py_plate(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Plate Notation")
    
    with expander: 
        st.code(py_plate)
    
    '''
    # Prior predictive checks
    
    '''
    ### plot ###
    selection_pp_pool = st.radio(
        "Choose prior level for prior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp_pool, col2_pp_pool = st.beta_columns(2)
    
    prior_pp_pool = translation_prior.get(selection_pp_pool)

    with col1_pp_pool: 
        st.image(f"plots_python/{model_context}_{prior_pp_pool}_prior_pred.jpeg")
        
    with col2_pp_pool: 
        st.image(f"plots_R/{model_context}_{prior_pp_pool}_prior_pred.png")
        
    ### code ###
    py_pp = ct.py_pp(model_name, idata_name)
    R_pp = ct.R_pp(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Prior predictive checks")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_pp)
        with col2:
            st.code(R_pp) 
    
    '''
    # Sample posterior

    '''
    
    ### code ###
    py_sample = ct.py_sample(model_name, idata_name)
    R_sample = ct.R_sample(model_name, model_formula, model_family, prior_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Sample posterior")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_sample)
        with col2:
            st.code(R_sample) 
    
    
    '''
    # Check traces (sampling)
    '''
    
    ### plot ###
    selection_trace_pool = st.radio(
        "Choose prior level for trace plot", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_trace_pool, col2_trace_pool = st.beta_columns(2)
    
    prior_trace_pool = translation_prior.get(selection_trace_pool)
    
    with col1_trace_pool: 
        st.image(f"plots_python/{model_context}_{prior_trace_pool}_plot_trace.jpeg")
    with col2_trace_pool: 
        st.image(f"plots_R/{model_context}_{prior_trace_pool}_plot_trace.png")
    
    ### code ###
    py_trace = ct.py_trace(idata_name)
    R_trace = ct.R_trace(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Trace-plot")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_trace)
        with col2:
            st.code(R_trace) 
    
    '''
    # Summary
    
    '''
    
    ### plot ###
    selection_summary = st.radio(
        "Choose prior level for summary", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_summary = translation_prior.get(selection_summary)
    
    st.image(f"plots_python/{model_context}_{prior_summary}_summary.png")
    
    ### code ###
    py_summary = ct.py_summary(idata_name)
    R_summary = ct.R_summary(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Summary")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_summary)
        with col2:
            st.code(R_summary) 

    '''
    # Posterior Predictive checks 
    
    '''
    
    ### plot ###
    selection_pp2_pool = st.radio(
        "Choose prior level for posterior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp2_pool, col2_pp2_pool = st.beta_columns(2)
    
    prior_pp2_pool = translation_prior.get(selection_pp2_pool)
    
    with col1_pp2_pool:
        st.image(f"plots_python/{model_context}_{prior_pp2_pool}_posterior_pred.jpeg")
        
    with col2_pp2_pool:
        st.image(f"plots_R/{model_context}_{prior_pp2_pool}_posterior_pred.png")
        
    ### code ###
    py_pp2 = ct.py_post_pred(model_name, idata_name)
    R_pp2 = ct.R_post_pred(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Posterior predictive")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_pp2)
        with col2:
            st.code(R_pp2) 

    '''
    # HDI (vs. data)
    
    '''
    
    ### plot ### 
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
        st.image(f"plots_python/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.jpeg")
        
    with col2_hdiplot:
        st.image(f"plots_R/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.png")
    
    ### code ###
    R_function, R_type = translation_code.get(selection_hdi2)
    
    if hdi_type == "fixed": 
        py_hdi = ct.py_hdi_data_fixed(hdi_type, idata_name)
        R_hdi = ct.R_hdi_fixed_groups(model_name, R_type, data_type, R_function, hdi_type)
    elif hdi_type == "full": 
        py_hdi = ct.py_hdi_data_full(hdi_type, idata_name)
        R_hdi = ct.R_hdi_full_groups(model_name, R_type, data_type, R_function, hdi_type)
    
    expander = st.beta_expander("🐒 Code-Monkey: HDI prediction intervals")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_hdi)
        with col2:
            st.code(R_hdi) 
            
    '''
    # HDI (parameters)
    
    '''
    
    ### plot ###
    selection_param = st.radio(
            "Choose prior level for HDI parameter plots", 
            ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
            index = 1)
    
    col1_hdi_param, col2_hdi_param = st.beta_columns(2)
    prior_param = translation_prior.get(selection_param)
    
    with col1_hdi_param: 
        st.image(f"plots_python/{model_context}_{prior_param}_HDI_param.jpeg")
    
    with col2_hdi_param: 
        st.image(f"plots_R/{model_context}_{prior_param}_HDI_param.png")

    ### code ###
    py_hdi_param = ct.py_hdi_param(idata_name)
    R_hdi_param = ct.R_hdi_param(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: HDI parameter intervals")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_hdi_param)
        with col2:
            st.code(R_hdi_param) 
    
elif choice == "Multilevel Covariation (model 3)":
    
    # for f-strings.
    model_context = "covariation"
    model_name = "m_covariation" # should be in args
    model_formula = "f_covariation" # should be in args
    model_family = "gaussian" 
    prior_name = "prior_covariation" # should be in args
    data_type = "train"
    idata_name = "covariation_idata"
    
    '''
    # Candidate model 3 (Random intercepts and slopes with covariation)
    Our third candidate model will be a multilevel model with both random
    intercepts and random slopes. The model will also model the covariance
    of the random intercepts and random slopes. This is almost always desirable,
    and is what happens by default in brms when we specify the (1+t|idx) formula. 
    In pyMC3 we have to build this ourselves of course. There is still a lot I 
    don't understand with regards to this, so do correct me & check the example docs.
    
    for R/brms, check [ajkurz CH13](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/adventures-in-covariance.html)
    
    for python/pyMC3, check [pyMC3 docs](https://docs.pymc.io/notebooks/multilevel_modeling.html)
    
    '''
    
    ### code ###
    py_reproducibility = ct.py_reproducibility()
    R_reproducibility = ct.R_reproducibility()
    
    expander = st.beta_expander("🐒 Code-Monkey: Packages & Reproducibility")
    
    with expander: 
        col1_reproducibility, col2_reproducibility = st.beta_columns(2)
        with col1_reproducibility:
            st.code(py_reproducibility)
        with col2_reproducibility: 
            st.code(R_reproducibility) 
    
    ### code ###
    py_preprocessing = ct.py_preprocessing()
    R_preprocessing = ct.R_preprocessing()
    
    expander = st.beta_expander("🐒 Code-Monkey: Preprocessing")
    
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1:
            st.code(py_preprocessing)
        with col2: 
            st.code(R_preprocessing) 
    
    '''
    # Model specification (math)
    
    Here, we specify our new model with random intercepts and slopes,
    as well as the covariance/correlation distributed as LKJ.
    For a more thorough explanation and reference, see [ajkurz CH 13](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/adventures-in-covariance.html#varying-slopes-by-construction)
    
    '''
    
    st.latex(r''' 
        y_{i, j} \sim Normal(\mu_{i, j}, \sigma) \\
        \mu_{i, j} = \alpha_{idx_j} + \beta_{idx_j} \cdot x_i \\
        \begin{bmatrix} \alpha_{idx} \\ \beta_{idx} \end{bmatrix} \sim MvNormal \left( \begin{bmatrix} \alpha \\ \beta \end{bmatrix}, \mathbf{S} \right) \\
        \mathbf{S} = \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix} \mathbf{R} \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix} \\ 
        \alpha \sim Normal(1.5, 0.5) \\
        \beta \sim Normal(0, 0.5) \\
        \sigma \sim HalfNormal(0.5) \\
        \sigma_{\alpha} \sim HalfNormal(0.5) \\
        \sigma_{\beta} \sim HalfNormal(0.5) \\
        \mathbf{R} \sim LKJcorr(2)
        ''')
    
    '''
    Where **S** is the covariance matrix and **R** is the corresponding correlation matrix. 
    **R** is distributed as **LKJcorr(2)**.
    Again, see [ajkurz CH13](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/adventures-in-covariance.html#varying-slopes-by-construction)
    '''
    
    '''
    # Model specification (code)
    
    As you will notice our code translation is not in both cases a one-to-one mapping.
    It is pretty complicated, and the implementation which appears to be most common
    in pyMC3 deviates slightly. As we will see from our plots and inference however, the
    two models we create (in brms and pyMC3) appear to be more or less identical. 
    
    '''
    
    ### code prep ###
    selection_prior = st.radio(
        "Choose prior level (how uncertain)", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_choice = translation_sigma.get(selection_prior)
    
    ### code ###
    py_model = ct.py_covariation(model_name, prior_choice)
    r_model = ct.R_covariation(model_name, model_formula, prior_name, prior_choice)

    expander = st.beta_expander("🐒 Code-Monkey: Model specification")
    with expander: 
        col1_model, col2_model = st.beta_columns(2)
        
        with col1_model:
            st.code(py_model)
        
        with col2_model: 
            st.code(r_model)
    
    '''
    # Plate Notation 
    
    '''
    
    ### plot ###
    st.image(f"plots_python/{model_context}_plate.png")
    
    ### code ###
    py_plate = ct.py_plate(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Plate Notation")
    
    with expander: 
        st.code(py_plate)
    
    '''
    # Prior predictive checks
    
    '''
    ### plot ###
    selection_pp_pool = st.radio(
        "Choose prior level for prior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp_pool, col2_pp_pool = st.beta_columns(2)
    
    prior_pp_pool = translation_prior.get(selection_pp_pool)

    with col1_pp_pool: 
        st.image(f"plots_python/{model_context}_{prior_pp_pool}_prior_pred.jpeg")
        
    with col2_pp_pool: 
        st.image(f"plots_R/{model_context}_{prior_pp_pool}_prior_pred.png")
        
    ### code ###
    py_pp = ct.py_pp(model_name, idata_name)
    R_pp = ct.R_pp(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Prior predictive checks")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_pp)
        with col2:
            st.code(R_pp) 
    
    '''
    # Sample posterior
    
    '''
    
    ### code ###
    py_sample = ct.py_sample(model_name, idata_name)
    R_sample = ct.R_sample(model_name, model_formula, model_family, prior_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Sample posterior")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_sample)
        with col2:
            st.code(R_sample) 
    
    
    '''
    # Check traces (sampling)
    '''
    
    ### plot ###
    selection_trace_pool = st.radio(
        "Choose prior level for trace plot", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_trace_pool, col2_trace_pool = st.beta_columns(2)
    
    prior_trace_pool = translation_prior.get(selection_trace_pool)
    
    with col1_trace_pool: 
        st.image(f"plots_python/{model_context}_{prior_trace_pool}_plot_trace.jpeg")
    with col2_trace_pool: 
        st.image(f"plots_R/{model_context}_{prior_trace_pool}_plot_trace.png")
    
    ### code ###
    py_trace = ct.py_trace(idata_name)
    R_trace = ct.R_trace(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Trace-plot")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_trace)
        with col2:
            st.code(R_trace) 
    
    '''
    # Summary
    
    '''
    
    ### plot ###
    selection_summary = st.radio(
        "Choose prior level for summary", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    prior_summary = translation_prior.get(selection_summary)
    
    st.image(f"plots_python/{model_context}_{prior_summary}_summary.png")
    
    ### code ###
    py_summary = ct.py_summary(idata_name)
    R_summary = ct.R_summary(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Summary")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_summary)
        with col2:
            st.code(R_summary) 

    '''
    # Posterior Predictive checks 
    
    '''
    
    ### plot ###
    selection_pp2_pool = st.radio(
        "Choose prior level for posterior predictive", 
        ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
        index = 1)
    
    col1_pp2_pool, col2_pp2_pool = st.beta_columns(2)
    
    prior_pp2_pool = translation_prior.get(selection_pp2_pool)
    
    with col1_pp2_pool:
        st.image(f"plots_python/{model_context}_{prior_pp2_pool}_posterior_pred.jpeg")
        
    with col2_pp2_pool:
        st.image(f"plots_R/{model_context}_{prior_pp2_pool}_posterior_pred.png")
    
    ### code ###
    py_pp2 = ct.py_post_pred(model_name, idata_name)
    R_pp2 = ct.R_post_pred(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Posterior predictive")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_pp2)
        with col2:
            st.code(R_pp2) 

    '''
    # HDI (vs. data)
    
    '''
    
    ### plot ### 
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
        st.image(f"plots_python/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.jpeg")
        
    with col2_hdiplot:
        st.image(f"plots_R/{model_context}_{prior_level_hdi}_HDI_{hdi_type}.png")
    
    ### code ###
    R_function, R_type = translation_code.get(selection_hdi2)
    
    if hdi_type == "fixed": 
        py_hdi = ct.py_hdi_data_fixed(hdi_type, idata_name)
        R_hdi = ct.R_hdi_fixed_groups(model_name, R_type, data_type, R_function, hdi_type)
    elif hdi_type == "full": 
        py_hdi = ct.py_hdi_data_full(hdi_type, idata_name)
        R_hdi = ct.R_hdi_full_groups(model_name, R_type, data_type, R_function, hdi_type)
    
    expander = st.beta_expander("🐒 Code-Monkey: HDI prediction intervals")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_hdi)
        with col2:
            st.code(R_hdi) 
            
    '''
    # HDI (parameters)
    
    '''
    
    ### plot ###
    selection_param = st.radio(
            "Choose prior level for HDI parameter plots", 
            ("Weak (sd = 5)", "Generic (sd = 0.5)", "Specific (sd = 0.05)"),
            index = 1)
    
    col1_hdi_param, col2_hdi_param = st.beta_columns(2)
    prior_param = translation_prior.get(selection_param)
    
    with col1_hdi_param: 
        st.image(f"plots_python/{model_context}_{prior_param}_HDI_param.jpeg")
    
    with col2_hdi_param: 
        st.image(f"plots_R/{model_context}_{prior_param}_HDI_param.png")

    ### code ###
    py_hdi_param = ct.py_hdi_param(idata_name)
    R_hdi_param = ct.R_hdi_param(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: HDI parameter intervals")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_hdi_param)
        with col2:
            st.code(R_hdi_param) 

elif choice == "Model Comparison":
    
    prior_context = "generic"
    data_type = "train"
    
    '''
    # Model comparison
    
    We will only now compare the models with what I have
    
    called "generic" priors (sigma, sd = 0.5). 
    
    I like to compare models in two ways: 
    
    1. Using information criteria (loo). 
    
    2. Using posterior predictive checks (& potentially predictions on unseen data). 
    
    If you want to know more check the 🧙‍♂️ Concept-Guru section. 
    
    '''
    
    expander = st.beta_expander("🧙‍♂️ Model Comparison")
    
    with expander: 
        
        '''
        The second one always works (pp checks). Since Bayesian models are always generative (see McElreath)
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
    
    ### plot ### 
    selection_pp = st.radio(
        "Choose model type to display posterior predictive checks for ", 
        ("Pooled", "Intercept", "Covariation"),
        index = 1)
    
    col1_pp, col2_pp = st.beta_columns(2)
    
    model_pp = translation_model.get(selection_pp)

    with col1_pp: 
        st.image(f"plots_python/{model_pp}_{prior_context}_posterior_pred.jpeg")
        
    with col2_pp: 
        st.image(f"plots_R/{model_pp}_{prior_context}_posterior_pred.png")
    
    
    ### code ###
    model_name = translation_mod_name.get(selection_pp)
    idata_name = translation_idata.get(selection_pp)
    py_trace = ct.py_trace(idata_name)
    R_trace = ct.R_trace(model_name)
    
    expander = st.beta_expander("🐒 Code-Monkey: Trace-plot")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_trace)
        with col2:
            st.code(R_trace) 
    
    '''
    # Compare HDI
    
    Now lets look at model predictions,
    both for fixed effects only, and with the full uncertainty. 
    Again: which model do you think we should prefer?
    
    '''
    
    ### plot ###
    col1_hdi1, col2_hdi1 = st.beta_columns(2)
    
    with col1_hdi1:
        selection_hdi1 = st.radio(
            "Choose model type to display HDI prediction intervals for", 
            ("Pooled", "Intercept", "Covariation"),
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
        st.image(f"plots_python/{model_hdi}_{prior_context}_HDI_{uncertainty}.jpeg")
        
    with col2_hdi2: 
        st.image(f"plots_R/{model_hdi}_{prior_context}_HDI_{uncertainty}.png")
    
    ### code ###
    model_name = translation_mod_name.get(selection_hdi1)
    R_function, R_type = translation_code.get(selection_hdi2)
    idata_name = translation_idata.get(selection_hdi1)
    
    if uncertainty == "fixed": 
        py_hdi = ct.py_hdi_data_fixed(uncertainty, idata_name)
        R_hdi = ct.R_hdi_fixed_groups(model_name, R_type, data_type, R_function, uncertainty)
    elif uncertainty == "full": 
        py_hdi = ct.py_hdi_data_full(uncertainty, idata_name)
        R_hdi = ct.R_hdi_full_groups(model_name, R_type, data_type, R_function, uncertainty)
    
    expander = st.beta_expander("🐒 Code-Monkey: HDI prediction intervals")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_hdi)
        with col2:
            st.code(R_hdi) 
    
    '''
    # Information criterion (loo)
    
    We have now done some eye-balling and will now check what loo has to say. 
    
    '''
    
    st.image(f"plots_python/loo_comparison.png")
    
    '''
    add "Concept-Guru", explaining that loo is leave-one-out approximation. 
    Here it is indicated that pooled model "underfits" and that
    student-t model "overfits" (i.e. the extra parameter (nu "v"))
    is not giving us enough to earn its keep. 
    
    '''

elif choice == "Prediction": 
    
    model_name = "m_covariation"
    idata_name = "covariation_idata"
    data_type = "test"
    
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
    
    ### code ###
    py_prep = ct.py_pred_prep(model_name, idata_name)
    R_prep = ct.R_pred_prep()
    
    expander = st.beta_expander("🐒 Code-Monkey: preprocessing & sampling")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_prep)
        with col2:
            st.code(R_prep) 
    
    '''
    
    # HDI prediction interval 
    
    NB: (perhaps we should at least have both mean forecast and individual forecast).
    
    '''
    
    ### plot ###
    col1, col2 = st.beta_columns(2)
    
    with col1: 
        st.image("plots_python/covariation_generic_HDI_predictions.jpeg") 
    
    with col2: 
        st.image("plots_R/covariation_generic_HDI_predictions.png")
    
    ### code ###
    # manually set for now. 
    uncertainty = "full"
    R_type = ".prediction" 
    R_function = "add_predicted_draws" 
    
    py_pred = ct.py_hdi_data_full(uncertainty, idata_name)
    R_pred = ct.R_hdi_full_groups(model_name, R_type, data_type, R_function, uncertainty)
    
    expander = st.beta_expander("🐒 Code-Monkey: HDI prediction intervals")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_pred)
        with col2:
            st.code(R_pred) 
    
    
    '''
    Should have some overthinking box with both (1) 
    shared variables and (2) that prediction is basically
    the same that we have been doing the whole time..
    New data does not really alter the process. We still
    just use our estimated parameters and our likelihood
    function to generate predictions (just as with the 
    posterior predictive). 
    '''
    
elif choice == "References & Inspiration":
    
    '''
    # Bayesian statistics
    
    * Collection of good books: https://docs.pymc.io/learn.html
    '''
    
    '''
    # python/pyMC3: 
    
    * Collection of pyMC3 ressources, including recodings of Rethinking Statistics (McElreath): https://github.com/pymc-devs/resources
    * Example notebooks (some more advanced analyses): https://docs.pymc.io/nb_examples/index.html
    '''
    
    '''
    # R/brms: 
    
    * Statistical Rethinking (McElreath) recoded in brms by Solomon Kurz: https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/
    * List of blogposts (some advanced analyses): https://paul-buerkner.github.io/blog/brms-blogposts/
    '''
