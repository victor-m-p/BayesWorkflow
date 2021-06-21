# import packages
from google.protobuf.reflection import ParseMessage
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
    'Pooled': 'idata_pooled',
    'Intercept': 'idata_intercept',
    'Covariation': 'idata_covariation'
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
    
    Please get in touch if you find the app interesting, have suggestions for improvements, hit upon bugs or would like to see 
    something covered which is not implemented at present. The app is a work in progress (and will probably stay that way) so I 
    am eager to receive feedback to further develop this ressource.
    
    '''
    
    
    '''
    # Purpose 
    
    This notebook attempts to show how to carry out two (almost) identical bayesian analyses, using **pyMC3** (python) and **brms** (R). 
    The main purpose of this app is to show users with experience in one language how to carry out a bayesian analysis in the other. 
    I have specifically had people who are looking to transition from brms to pyMC3 in mind, but people coming from pyMC3 should also find the app interesting.
    I have tried to explain the code and concepts in selected parts, but this is not supposed to be an exhaustive guide to bayesian statistics nor pyMC3 or brms. 
    If you want to dive deeper into either bayesian statistics or more advanced analyses I have provided a list of good resources in the appendix (References & Inspiration).
    
    '''
    
    '''
    # How to use 
    
    For all parts of the analysis (for both languages) you can access reproducible code, and easily copy it to clipboard.
    I hope that this will encourage you to run the code alongside the app, since this is the only way to really understand what is going on.
    The app is meant to be followed sequentially as a cohesive introduction & the code in later parts rely on earlier parts of the pipeline.
    While building a bridge between pyMC3 and brms is the main objective, I hope that parts of the analysis & workflow might also lead you to a better Bayesin workflow.
    
    '''
    
    '''
    # Functionality 
    
    You will see boxes with the titles **Code-Monkey**, **Language-Learner** and **Concept-Guru**. These let you dive deeper into the material:
    
    * :monkey: **Code-Monkey**: Display code to reproduce analysis
    * :keyboard: **Language-Learner**: Explanations of code-implementation differences between python/pyMC3 & R/brms
    * :male_mage: **Concept-Guru**: Conceptual deep-dives

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
    
    Based on 15 consecutive years of data (t), corresponding gradings (y) and a sample of 15 aliens (ID) 
    we want to infer how fast how aliens learn about danish culture (slope/beta), and how much they know when they arrive (intercept/alpha). 
    Notice that our true *alpha* is equal to 1 and our true *beta* is equal to 0.3. These are the underlying values that we would ideally
    like our model to be able to infer from the data.
    We might also be interested in the variability between aliens or our left-over uncertainty, but let's table that for now. 
    Remember to check out the :monkey: *Code-Monkey* boxes to follow along with the code. 
    
    '''
    
    ### code ###
    expander = st.beta_expander("🐒 Code-Monkey: Reproducibility")
    py_reproducibility = ct.py_reproducibility()
    R_reproducibility = ct.R_reproducibility()
    
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1:
            st.code(py_reproducibility)
        with col2: 
            st.code(R_reproducibility)
        
    
    ### code ##
    expander = st.beta_expander('🐒 Code-Monkey: Simulation')
    py_sim = ct.py_sim()
    R_sim = ct.R_sim()
    
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_sim)
        with col2: 
            st.code(R_sim)
            
    '''
    # Quick EDA
    
    Below is a scatter plot of the *training* data, where a regression line is showed for each alien (ID). 
    We use the data generated in python for all subsequent analysis. 
    
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
    idata_name = "idata_pooled"
    
    '''
    # Candidate model 1 (Complete Pooling)
    
    Our first candidate model will be a complete pooling model. 
    This model treats each observation at each time-point as if it belongs to the same alien (ID),
    and will only estimate one intercept (alpha) and one slope (beta). 
    You might already feel that this is not a satisfactory model, but bear with me for now.
    We will build on top of what we have learned and get to more complex models in the next sections. 
    
    '''

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
    
    expander = st.beta_expander("⌨️ Language-Learner: Coords & dims?")
    
    with expander: 
        '''
        The preprocessing for the pyMC3 analysis is more extensive than the
        proprocessing for the brms analysis. This is because we manually
        specify the shape of our data, whereas this is handled
        for us automatically in brms. 
        
        # Coords & dims 
        I like to use labelled coords and dims instead of working with 
        unlabelled arrays (which is also possible). It makes the whole analysis
        pipeline a lot easier to handle if we invest some time in gathering our
        data in the proper format before we start modeling. To really get familiar
        with how to use labelled coords and dims for pyMC3 analyses check this
        [great tutorial](https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/xarray/2020/09/22/pymc3-arviz.html).
        
        # Python
        If the python code looks unfamiliar I suggest checking out core libraries such
        as [numpy](https://numpy.org/) and concepts such as [dictionaries](https://realpython.com/python-dicts/)
        before proceeding. For bayesian analysis with pyMC3 and Arviz specifically, some knowledge
        of [xarray](http://xarray.pydata.org/en/stable/) data structures is also helpful. 
        '''
    
    '''
    # Model specification (math)
    
    We can formulate the complete pooling model (with generic priors).
    
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
    I have run all analysis for three levels of priors. A *generic prior* (sigma and sd = 0.5)
    is shown by default, with the option of displaying a *specific prior* (sigma and sd = 0.05) 
    or a *weak prior* (sigma and sd = 5).
    For more on priors, see [Gelman's prior choice recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations).
    
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
            
            # Model formula vs. manual specification
            
            In brms we specify our model via a model formula (to the right in the expandable section above). In pyMC3 we can either explicitly specify our model,
            or we can do it via a model formula (implemented in the ```glm``` module). I will use the explicit notation here, because it 
            forces me to understand how the model works and is more flexible. The model formula might seem more intuitive or user friendly, 
            but I think that it is easy to go wrong when we do not know what is happening under the hood. 
            
            for example, notice that we have specified the intercept with the special ```0 + Intercept``` syntax, rather than the more common ```1 + ... ``` syntax. 
            We should do this if we have not mean-centered our predictors (which brms expects) and are creating a model that is not an intercept only model (ours is not).
            Something like this is easy to miss (which I actually did at first). You can read more about the ```0 + Intercept``` syntax in [ajkurz CH15](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/horoscopes-insights.html#use-the-0-intercept-syntax).
            
            # Shared Variables 
            
            In the pyMC3 code (left) we have created something called a *shared variable* for our x variable, time (t) with the ```pm.Data()``` function.
            This has to be done if we want to generate predictions based on new data (data that the model was not trained on).
            We only need to create a shared variable for t here, because it is the only predictor variable in the model. 
            pyMC3 relies on **theano** as the backend (just as brms relies on **stan**), and theano needs to encode variables that we might want to change later in a different format. 
            
            # Sigma (Normal or HalfNormal?) 
            
            We have specified a *Normal distribution* for the sigma parameter in the brms priors (right), but we have specified a *Half-normal distribution* for the sigma parameter i pyMC3 (left). 
            This might seem confusing, but the prior for the sigma distribution cannot actually be a normal distribution (as this would allow for negative values). 
            The reason why we can do this in brms is because it realizes that draws from the sigma distribution have to be positive. 
            So, when it encounters a negative value it throws away that sample and tries again. This will then effectively give us a Half-normal distribution. 
            pyMC3 does not baby-sit in this way, and will throw an error if we try to specify a normal distribution for the sigma parameter. 
            You can try this for yourself and see the machinery break down! 
            
            # The model as a function? 
            
            In pyMC3 I have specified the model inside a function. This might seem weird,
            and it is also not necessary. The benefit of doing it is that when we have finished
            modeling we can save our idata (to be introduced shortly) but we will still have to 
            recompile our model when we return to our project. At this point we can just re-run 
            the function that we have already specified. 
            '''
    
    '''
    # Plate Notation 
    
    Something that is really nice in pyMC3 is that we can check whether we specified our model as we intended to. 
    The ```model_to_graphviz()``` function from pyMC3 shows the model we have specified in plate notation, which is regrettably less intuitive than the awesome [Kruschke diagrams](http://www.sumsar.net/blog/2013/10/diy-kruschke-style-diagrams/).
    They basically communicate the same though, and once we learn to read them they are a great sanity check. Here we can see that we have estimated just one *alpha* distribution
    and one *beta* distribution, as well as an overall model error *sigma*. 
    In addition, we the model informs us that the data has shape $15 \cdot 10$ which
    is correct because we have $15$ aliens (ID) and $10$ time-points (t) for each. 
    
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
    as compared to the data. We can get nice *prior predictive checks* in both pyMC3 with and brms.
    In pyMC3 we use the ```plot_ppc()``` function from the **Arviz** plotting library, and in brms we rely on the ```pp_check()``` function.
    Both ```plot_ppc()``` and ```pp_check()``` are used for both prior predictive checks and posterior predictive checks
    (depending on the data that we input to the functions).  
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
    
    expander = st.beta_expander("⌨️ Language-Learner: Prior predictive checks")
    
    with expander: 
        '''
        # brms 
        In the brms code (right) we simply pass our model fitted with 
        ```sample_prior = "only"``` to the ```pp_check()``` function.
        The ```pp_check()``` function will visualize prior predictive draws
        because this is what is available (i.e. we have not sampled the posterior yet). 
        I am not sure whether it is possible to force ```pp_check()``` to show
        prior draws once we have sampled the posterior with ```sample_prior = TRUE```.
        Let me know if this is indeed possible. 
        
        # pyMC3
        In pyMC3 we have only compiled our model so far. Whereas model compilation & 
        sampling happens in the same step for brms, they are separate steps in pyMC3.
        This means that we now have to sample the prior predictive with the 
        ```sample_prior_predictive()``` function from pyMC3 before we can visualize
        draws from the prior predictive. 
        
        '''
        
    expander = st.beta_expander("🧙‍♂️ Concept-Guru: Prior predictive checks")
    
    with expander: 
        '''
        # Reasonable scale for priors
        A reasonable scale for our priors will ideally be informed by 
        our a priori knowledge of the phenomenon that we are modeling. 
        We might for instance know that plausible human heights range 
        somewhere between $1.5$ 
        and $2.5$ meters, and incorporate this
        knowledge in to our model by specifying priors which assign most
        probability density to this region of observations. 
        
        In our case of aliens' knowledge of danish it is slightly more
        tricky, and we might want to inspect our data to figure out what
        a reasonable range is (if we assume that we do not have any prior
        knowledge). We almost always know something however, and 
        say that we know the danish test scores knowledge between 
        $-5$ and $+10$ points (which is consistent with our observations). 
        Then the prior predictive check for the "generic" priors is 
        reasonable, placing most probability density in the $-1$
        to $+3$ range, but reserving some probability for more extreme 
        outcomes. The prior predictive check for the "weak" priors is
        too broad, with a range of $-100$ to $+100$, while the prior
        predictive check for the "specific" priors is too restrictive,
        placing almost all of the probability density in the region
        between $1$ and $2$. 
        
        In many cases (when we have enough observations) it is not 
        too problematic to have vague (or "weak") priors, and you
        will generally notice that the model with "weak" priors 
        infers parameter values that are close to the model fitted 
        with "generic" priors. However, the "specific" priors should
        only be used if we have strong reason to bias the model
        (e.g. based on a meta-analysis). You will see that the 
        parameter values that this model infers differ significantly from the
        parameter values of the two other models. 
        '''
    '''
    # Sample posterior
    
    We have now verified that the we have specified our model correctly with the plate diagram,
    and let's say that we are happy with our prior predictive checks.
    We should now press the inference button and sample the posterior. 
    
    '''
    
    ### code ###
    py_sample = ct.py_sample(model_name, idata_name)
    R_sample = ct.R_sample(model_name, model_formula, model_family, prior_name, model_context, prior_pp_pool)
    
    expander = st.beta_expander("🐒 Code-Monkey: Sample posterior")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_sample)
        with col2:
            st.code(R_sample) 
    
    expander = st.beta_expander("⌨️ Language-Learner: Inference button")
    
    with expander:
        '''
        # Overall structure
        As we noted in the *Language-Learner* section for prior predictive checks,
        the model compilation and model sampling steps are separated in pyMC3 
        whereas they are not in brms. This means that we now have to recompile our
        model in brms with ```sample_prior = TRUE``` whereas we continue to use
        our compiled model from pyMC3 and now sample the posterior (just as we 
        sampled the prior before). Notice that we keep adding our data objects
        (prior predictive, posterior, etc.) to our idata object, which I just find
        much easier than having one array of samples for each. This follows the 
        great workflow of [oriol](https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/xarray/2020/09/22/pymc3-arviz.html).
        
        # Tuning
        For all models in this tutorial, I will be using some model tuning
        settings that I have adopted from Riccardo Fusaroli. 
        Specifically, we set the ```target_accept``` (pyMC3) and 
        ```adapt_delta``` (brms) to .99 and the ```max_treedepth``` (pyMC3 & brms) to 20. 
        Both will mean that our samples will be more expensive (computationally)
        but they will also be of higher quality, and might save us time in the long
        run, by avoiding the need to re-run models with divergences. 
        
        '''
    
    '''
    # Check traces (sampling)
    
    The first thing we might want to check after sampling the posterior is wheather computation was successfull. 
    I like to generate *trace plots* at this point. In pyMC3 we can use the ```plot_trace()```
    function from **Arviz** and in brms we can use the ```plot()``` function from **brms**. 
    If we observe issues at this point (e.g. divergences, poor mixing, etc.) 
    there are good diagnostic tools available for both [pyMC3 models](https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html)
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
    
    For all prior levels we see healthy traces to the right (caterpillars),
    and we see reasonably smooth and well-mixed KDE/histograms to the left. 
    However, notice that the values for our parameters differ based on our priors.
    The models fitted with either *weak* or *generic* priors have estimated the
    parameters similarly, 
    but the model fitted with *specific* priors visibly restrict the posterior
    (i.e. the priors constrain the posterior unduly). One could use *updating checks*
    to visualize how constrained the posterior is by the prior. 
    '''
    
    '''
    # Summary
    
    We can now (optionally) check the summary of the model. We might not be interested in the estimated parameters (yet)
    but the summary also gives us information about the number of **effective samples** and **R-hat values**. 
    We can note already that the model with generic priors has inferred *alpha* to be 1.1 (vs. actual 1) and *beta* to be 0.2 (vs. actual 0.3).
    Note that our actual data might not have mean *alpha* equal to 1 and mean *beta* equal to 0.3. 
    
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
    
    If all is well so far we should now generate *posterior predictive checks*. Similarly to *prior predictive checks* we have blue draws and a black line
    showing the true distribution. We see that the model has not quite captured the shape of 
    the posterior distribution (e.g. the modes differ). We see that the posterior predictive checks look
    particularly bad for the *specific* prior. Here we have constrained the model with very strong priors,
    where almost all of the probability mass for the prior predictive check was in the $[1, 2]$ range. The model has learned from the
    data and now places the majority of its weight in the $[1, 3]$ range, but this is still not 
    ideal, unless we have a strong reason to bias the model this heavily. 
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
    
    ### language learner ###
    expander = st.beta_expander("⌨️ Language-Learner: Posterior predictive checks")
    
    with expander: 
        '''
        Just as with the prior predictive checks we need to sample the 
        posterior predictive before we can visualize it in pyMC3. This happens with
        the ```sample_posterior_predictive()``` function from pyMC3.
        '''
    
    ### concept guru ###
    expander = st.beta_expander("🧙‍♂️ Concept-Guru: Posterior predictive checks")

    with expander: 
        '''
        # What went wrong?
        We see that the mode of the true posterior distribution and the mode of our predictive draws differ systematically. 
        The posterior has long tails which is not well captured by our model. Posterior predictive checks are not in themselves
        sufficient to accept or reject a model. Sometimes we can reasonably approximate e.g. a non-Gaussian distribution with a 
        Gaussian, and sometimes even when the posterior predictive looks really good we could face other issues.
        However, generally they do provide a good idea about whether our model-structure is proper. 
        Often when the posterior predictive check is bad it is because we have either (1) not implemented a proper
        *random effects structure* or (2) not specified a proper *likelihood function*. 
            
        '''
    
    '''
    # HDI (vs. data)

    We can also now run the model forward (i.e. generate predictions from the model). 
    We can compare these model predictions with the data that the model is trained on, to check whether the model has captured the patterns in the data.
    We can do this either for (a) **fixed effects only** or (b) with the **full model uncertainty**. 
    If we generate predictions for fixed effects only, we will get predictions for the *mean* of the population. 
    If we generate predictions with the full model uncertainty  (incl. sigma and potential random effects) we will get predictions for individuals.
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
    
    # language learner #
    expander = st.beta_expander("⌨️ Language-Learner: Posterior Predictive (& HDI)")
    
    with expander: 
        '''
        # brms 
        In brms, we use the convenience functions ```add_fitted_draws()``` and 
        ```add_predicted_draws``` from the [tidybayes package](http://mjskay.github.io/tidybayes/articles/tidy-brms.html).
        The functions allow us to gather draws in a tidy format 
        for either fixed effects, or with the full uncertainty of the model.
        For an example of how to do this with both tidybayes (as here) and 
        manually in R, see [ajkurz CH12](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/multilevel-models.html).
        
        # pyMC3
        for the *fixed effects only* predictions in pyMC3 we extract 
        our posterior predictive draws of the *alpha* and *beta* parameters
        and multiply *beta* with our x-variable time (t). 
        For the *full uncertainty* predictions we simply use our y predictions
        from the posterior predictive. Note that we have to do a little bit
        of reshaping (data wrangling) to make the dimensions compatible. 
        
        '''
    # concept guru #
    expander = st.beta_expander("🧙‍♂️ Concept-Guru: HDI intervals")
    
    with expander: 
        '''
        Do you think the posterior predictive (and HDI intervals) do a good job of reproducing the data?
        
        Something we might note for the *fixed effects only* predictions is that the trend looks right,
        but that the HDI intervals are very narrow. Models that do not use the proper hierarchical structure
        of the data are typically over confident because they do not incorporate the proper uncertainty. 
        
        For the *full uncertainty* predictions we see that the HDI intervals appear sufficiently wide,
        but we also notice that the uncertainty intervals are constant throughout the time-points -
        even though we observe that the data-points (observations) are more spread out for the later time-points.
        The reason that we observe this is that the model has no parameter which allows it to become more uncertain
        for later time-points. The model only has three parameters; *alpha*, *beta* and *sigma* and we might
        now start to wonder how to improve the model to capture the fact that observations spread out with time. 

        '''
    
    '''
    # HDI (parameters)
    
    The last thing we might want to inspect at this point is the estimated distributions (and HDI intervals) for our inferred parameters. 
    In pyMC3 this can be achieved with the ```plot_forest()``` function from **Arviz** and in brms it can be achieved with the
    ```mcmc_areas()``` function from the **bayesplot** library. 
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
    
    expander = st.beta_expander("🧙‍♂️ Concept-Guru: inferred distributions")
    
    with expander: 
        '''
        The HDI intervals for our parameters *alpha*, *beta* and *sigma*
        are much narrower for the model fitted with a *specific* prior 
        than for models fitted with the *generic* and *weak* prior. 
        This is not because the model has good reason to be this sure about 
        our parameters given the data, but because we have specified extremely
        restrictive priors which force the model to be falsely certain about
        (wrong) parameter values. 
        '''
    
    '''
    # Save idata 
    
    Now we might want to make sure that we save our model (or samples)
    so that we can avoid sampling our model next time we open the project.
    In brms (as we have seen) we can save our model and with 
    ```file_refit = "on_change"``` brms will only recompile if we have made changes.
    
    In brms we manually save the **idata object** which contains all our samples
    (prior predictive, posterior, posterior predictive) and then recompile the
    model with the function we made earlier. 
    
    We will not do this here, but to check out ```arviz.to_netcdf()``` and 
    ```arviz.from_netcdf()``` for a convenient way to [save](https://arviz-devs.github.io/arviz/api/generated/arviz.to_netcdf.html)
    and [load](https://arviz-devs.github.io/arviz/api/generated/arviz.from_netcdf.html)
    **idata objects**. 
    
    '''
    
elif choice == "Random Intercepts (model 2)":
    
    # for f-strings.
    model_context = "intercept"
    model_name = "m_intercept" # should be in args
    model_formula = "f_intercept" # should be in args
    model_family = "gaussian" 
    prior_name = "prior_intercept" # should be in args
    data_type = "train"
    idata_name = "idata_intercept"
    
    r'''
    # Candidate model 2 (Random intercepts)
    Our second candidate model will extend on the complete pooling model to also include *random intercepts* ($\alpha$),
    for each alien (ID). It will thus be our first *multilevel* (*hierarchical*) model. 
    *Richard McElreath* argues in *Statistical Rethinking* that 
    "multilevel regression deserves to be the default form of regression". 
    Some often highlighted (e.g. *Richard McElreath*, *Riccardo Fusaroli*, *Osvaldo Martin*) advantages are:
    
    1. Accounting for repeated measures 
    2. Modeling individual-level effects 
    3. Partial Pooling (shrinkage) 
    
    
    No additional preprocessing is necessary (if you have been following the code
    in the earlier sections). 
    '''

    
    '''
    # Model specification (math) 
    
    Our new model can be formulated in mathematical (pseudo-code) notation as below.
    We are now estimating an intercept distribution for each alien (ID) and this
    distribution is based on an underlying distribution for the population.
    We now have two **hyper-priors** which will influence our intercept parameter (see e.g. *Bayesian Analysis with Python* by *Osvaldo Martin*)
    
    '''
    
    st.latex(r''' 
        y_{i, j} \sim Normal(\mu_{i, j}, \sigma) \\
        \mu_{i, j} = \alpha_{var_j} + \beta \cdot x_i \\
        \alpha_{var_j} \sim Normal(\alpha_j, \alpha_{\sigma_j}) \\
        \beta \sim Normal(0, 0.5) \\
        \alpha \sim Normal(1.5, 0.5) \\
        \alpha_{\sigma} \sim HalfNormal(0.5) \\
        \sigma \sim HalfNormal(0.5)
        ''')
    
    '''
    # Model specification (code)
    
    Again, we need to translate this into code in pyMC3 and brms. 
    
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
    
    expander = st.beta_expander("⌨️: Language-Learner: Model specification")
    
    with expander: 
        '''
        # Shared Variables (again)
        In the pyMC3 code (left) we now have *two* rather than one shared variable. 
        In addition to the x-variable time (t) we have now also specified ID as a shared variable.
        This is because ID is now a part of our model (random intercepts for IDs). 
        As such, if we wish to generate predictions for our IDs (or new IDs) we will want to
        be able to feed the model new data for this variable. 
        
        # Dims 
        Another thing to note in the pyMC3 code is that we have specified 
        ```dims = "idx"``` for the varying intercepts parameter. 
        This tells the model that we wish to estimate one distribution (parameter)
        for each of the IDs that we supplied in our ```coords``` earlier. 
        
        # Inheritance
        Note that the ```alpha_var``` parameter inherits *sigma* and *mu* from the population
        distributions. Each IDs intercept is drawn from these population distributions for 
        *mu* and *sigma*. This is less obvious from the brms implementation (right) but the 
        same thing is happening under the hood.  
        
        '''
        
    '''
    # Plate Notation 
    
    The structure of the model has changed as reflected in the plate diagram. 
    There is still only one *beta* and overall error *sigma* for the model. 
    However, we now estimate 15 *alpha* (intercept) values, one for each alien (ID). 
    These are shown to be drawn from the population level distributions (hyper-priors).
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
    
    Again, we now generate prior predictive checks. 
    The code to achieve this is the same as for the *complete pooling* model. 
    The prior predictive checks are also similar to those from earlier 
    in that the "generic" prior predictive checks look ok, while both the
    "weak" and the "specific" prior predictive checks are problematic
    (in different ways). 
    
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
    
    And again, we now press the inference button after having checked the plate diagram
    and found the prior predictive checks reasonable. The code to achieve this is the
    same as for the *complete pooling* model. 

    '''
    
    ### code ###
    py_sample = ct.py_sample(model_name, idata_name)
    R_sample = ct.R_sample(model_name, model_formula, model_family, prior_name, model_context, prior_pp_pool)
    
    expander = st.beta_expander("🐒 Code-Monkey: Sample posterior")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_sample)
        with col2:
            st.code(R_sample) 
    
    
    '''
    # Check traces (sampling)
    
    We follow the workflow from the first model (complete pooling) and check our traces.
    In the pyMC3 trace plot (left) we see that our ```alpha_var``` parameter shows us 
    the intercept distributions for each ID. This is the middle distribution with a lot of lines/KDEs.
    There are still no divergences for any of the models. So far so good. 
    
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
    
    Again, we produce a summary to check **effective samples** (good) and 
    **R-hat values** (good). We note that the estimation for our main parameters 
    (*alpha*, *beta*) has not drastically changed, but our overall model error 
    (*sigma*) has been reduced significantly as compared to the complete pooling model. 
    in pyMC3 (which is shown) we also get estimations for the inferred intercept (*alpha*)
    for each ID. 
    
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
    
    We now generate *posterior predictive checks*. 
    They now look slightly better than the complete pooling model
    that we did before, but we still notice the issue that the mode
    of the true posterior distribution and the mode of our posterior 
    predictive draws are systematically different. 
    
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
    
    We now see wider HDI intervals for the *fixed effects* only posterior predictions
    (for models fitted with *generic* and *weak* priors). This appears like a more 
    reasonable uncertainty estimation (around the population mean) than what we had
    for the *complete pooling* model. 
    
    Notice that the HDI intervals are still equally wide for all time-points (t). 
    We still have not given the model room to accommodate the fact that our observations
    are more spread out for later time-points than early time-points. We will tackle 
    this issue in the next section (*multilevel covariation*).
    
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
    
    We observe chiefly that our uncertainty with regards to *alpha*
    has now grown, while our model error *sigma* has been reduced. 
    The *random intercepts* model appears to be more reasonable than the
    *complete pooling* model (which it also should given our data generating process). 
    
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
    
    '''
    # Save idata 
    
    Again we could save our **Arviz** idata object at this point,
    before moving on to the next candidate model. 
    
    '''
        
elif choice == "Multilevel Covariation (model 3)":
    
    # for f-strings.
    model_context = "covariation"
    model_name = "m_covariation" # should be in args
    model_formula = "f_covariation" # should be in args
    model_family = "gaussian" 
    prior_name = "prior_covariation" # should be in args
    data_type = "train"
    idata_name = "idata_covariation"
    
    '''
    # Candidate model 3 (Random intercepts and slopes with covariation)
    Our third candidate model will be a multilevel model with both *random
    intercepts* and *random slopes*. The model will also model the *covariance*
    of the random intercepts and random slopes. This is almost always better than
    modeling random slopes and intercepts independently, 
    and is what happens by default in brms when we use the ```(1+t|idx)``` syntax in the model formula. 
    In pyMC3 we have to build this ourselves of course. 
    
    *NB: There is still a lot I don't understand with regards to this, 
    both mathematically and implementation wise. 
    Do correct me & check the [example docs for pyMC3](https://docs.pymc.io/notebooks/multilevel_modeling.html) 
    and [ajkurz for brms](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/adventures-in-covariance.html) for yourselves!*
    
    '''
    
    ### code ###
    py_preprocessing = ct.py_preprocessing_cov()
    
    expander = st.beta_expander("🐒 Code-Monkey: Additional preprocessing (python)")
    
    with expander: 
        st.code(py_preprocessing)

    
    '''
    # Model specification (math)
    
    Here, we specify our new model with random intercepts and slopes,
    as well as the covariance/correlation distributed as LKJ.
    LKJ(1) is a flat prior (all correlations equally likely) but I will stick to the brms/stan baseline recommendation
    of using an LKJ(2) prior (where extreme correlations are treated as less likely). 
    For a more thorough explanation and reference, see [ajkurz CH 13](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/adventures-in-covariance.html#varying-slopes-by-construction).
    
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
    Again, see [ajkurz CH13](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/adventures-in-covariance.html#varying-slopes-by-construction).
    '''
    
    '''
    # Model specification (code)
    
    As you will notice our code translation is not in both cases a one-to-one mapping of the pseudo-code.
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
    
    This is getting more and more interesting. 
    We now estimate *random intercepts* and *random slopes* as well as their *covariation* (LKJCholeskyCov).
    Notice that the *alpha_beta* parameter now has dimension $15 \cdot 2$ because we have $15$ random slopes
    and $15$ random intercepts (one for each participant). 
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
    
    Run prior predictive checks. The code is the same as for both previous models. 
    Because we have been increasing the amount of parameters in the model, we will by chance sample increasingly 
    extreme positive and negative values (there are many possible combinations of parameter values,
    some of which are extreme). 
    However, most of the probability density is still in the same regions as for previous models. 
    
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
    
    Press the inference button and sample!
    The code is the same as for both previous models,
    but if you are running the code yourself, be prepared
    that this model will take longer to sample from.
    If it is excessively slow, you could drop the tuning
    that we have been doing.
    
    '''
    
    ### code ###
    py_sample = ct.py_sample(model_name, idata_name)
    R_sample = ct.R_sample(model_name, model_formula, model_family, prior_name, model_context, prior_pp_pool)
    
    expander = st.beta_expander("🐒 Code-Monkey: Sample posterior")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_sample)
        with col2:
            st.code(R_sample) 
    
    
    '''
    # Check traces (sampling)
    
    We still do not get any divergences or observe any obvious issues.
    Notice that we have been tuning our models for both pyMC3 and brms (when sampling). 
    In particular we have set ```adapt_delta``` (pyMC3) and ```target_accept``` (brms) to .99
    and ```max_treedepth``` to 20 (pyMC3 & brms). This is more costly (slow) than
    relying on the baseline settings in either implementation, but gives us better samples,
    and helps ensure that the models will diverge less (given that our specification is reasonable).
    Especially the model with *specific* priors could give divergences without tuning
    (in fact I think it does, but try it out for yourself).
    
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
    
    Again, we observe that the estimation for our main parameters of interest, *alpha* and *beta*
    remain largely unchanged (although there is more uncertainty for both *alpha* and *beta* now).
    Our model error *sigma* is lower than for both the previous models.  
    
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
    
    WOW! The posterior predictive checks are starting to look really good 
    for the models fitted with *generic* and *weak* priors. 
    The models now reliably capture the shape of the posterior, and the 
    mode of the true posterior and the posterior predictive draws are now
    almost identical. 
    
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
    
    The posterior predictive for both *fixed effects* and *full uncertainty* 
    now both show the really nice property that the HDI intervals are 
    getting wider over time. 
    
    In addition, the HDI intervals are wider than for both the *complete pooling*
    and the *random intercepts* models. Our model is generating better predictions
    while also being more honest (uncertain) with regards to its capabilities. 
    
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
    
    Our model now has the right estimation for *sigma*, which we specified to be distributed as Normal(0, 0.5) in the simulation. 
    It also has most uncertainty with regards to the *alpha* parameter, which is
    also true to the data generating process (see Simulation & EDA). 
    *beta* is slightly low, and *alpha* is slightly high, but this could be due
    to random sampling variation in our simulation. 
    
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

    '''
    # Save idata 
    
    We could now save our **Arviz** idata object for later. 
    
    '''
        
elif choice == "Model Comparison":
    
    prior_context = "generic"
    data_type = "train"
    
    '''
    # Model comparison
    
    In this section we will do model comparison of the three kinds of models we have 
    implemented so far; *complete pooling*, *random intercepts* and *multilevel covariation* models.
    We will only compare models fitted with *generic* priors to keep it relatively clean. 
    
    I like to compare models in two ways: 
    
    1. Using information criteria (LOO). 
    
    2. Using posterior predictive checks and posterior predictive against data. 
    
    '''
    
    expander = st.beta_expander("🧙‍♂️ Concept-Guru: Model Comparison")
    
    with expander: 
        
        '''
        # Posterior Predictive 
        We can always evaluate model quality based on posterior predictive checks,
        and based on how the data looks in the predictive posterior (i.e. the HDI plots we have been doing).
        Since Bayesian models are always generative 
        we can simulate/generate new data based on our estimated distributions over parameters
        and our likelihood function (see *Statistical Rethinking* by *Richard McElreath*). 
        We should be able to (forward) generate data which captures the main patterns in the data that 
        the model is trained on. 
        
        # Information Criteria 
        Information criteria attempt to approximate cross-validation results.
        LOO-CV approximates leave-one-out cross validation, without actually 
        performing *k* iterations (see *Bayesian Analysis with Python* by *Osvaldo Martin*). 
        LOO is generally preferred over e.g. WAIC and is the baseline implementation. 
        An issue with this kind of model comparison is that it is typically not valid
        for comparing models with different *likelihood-functions* (a notable exception
        is that it is valid for comparing models with *Gaussian* and *Student-t* 
        likelihood functions). Since we have stuck with a *Gaussian* likelihood-function throughout
        it is not an issue in our case. 
        
        '''

    '''
    # Compare posterior predictive
    
    From visually inspecting (and comparing) the posterior predictive checks,
    the *multilevel covariation* model clearly reproduces the posterior shape much better
    than both the *complete pooling* and the *random intercepts* models. 
    
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
    
    Here we observe (as we had already noted earlier) that only the 
    *covariation* model seems to incorporate appropriate uncertainty,
    while also capturing the essential pattern that the data is more
    spread out for later time-points (t). 
    
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
    # Information criterion (LOO)
    
    Indeed, when we compare the models using LOO, the *covariation* model
    is heavily favored against both of the other models we have created. 
    This suggests that the additional parameters that we have estimated are
    supported by the data (they do not overfit). in the ```weight``` column,
    we can see that the *covariation* model gets almost all of the weight
    (indicating that our LOO comparison is very clearly in favor of this model). 
    Only python (*Arviz*) output shown below.
    
    '''
    
    st.image(f"plots_python/loo_comparison.png")
    
    # get the loo for python & R
    R_loo = ct.R_loo()
    py_loo = ct.py_loo()
    
    expander = st.beta_expander("🐒 Code-Monkey: LOO information criteria")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_loo)
        with col2:
            st.code(R_loo) 

elif choice == "Prediction": 
    
    model_name = "m_covariation"
    idata_name = "covariation_idata"
    data_type = "test"
    model_context = "covariation"
    idata_name = "idata_covariation"
    
    '''
    # Prediction on unseen data
    
    This will be only a brief introduction to prediction in *pyMC3* and *brms*.
    We will only cover predicting (1) groups that are already in the data
    and (2) only the trend for the population (not clusters or IDs). 
    If you want to see more on prediction (e.g. on new groups or on IDs)
    Then please let me know. The process is largely the same though, and 
    you can check the "References & Inspiration" page to see where to go next. 
    In this section we will only be predicting from the *covariation* model,
    as we previously found that this was our most appropriate model. 
    
    '''

    '''
    # Preprocessing 
    
    We need some additional preprocessing for this step in pyMC3,
    whereas we just need to make sure that we have the *test* data
    loaded in brms. 
    
    '''
    
    ### code ###
    py_prep2 = ct.py_pred_prep(model_name, idata_name)
    R_prep2 = ct.R_pred_prep()
    
    expander = st.beta_expander("🐒 Code-Monkey: Preprocessing (test data)")
    with expander: 
        col1, col2 = st.beta_columns(2)
        with col1: 
            st.code(py_prep2)
        with col2:
            st.code(R_prep2) 
    
    expander = st.beta_expander("⌨️ Language-Learner: Prediction & Shared variables")
    
    with expander: 
        '''
        in pyMC3 we finally have an excuse to use the shared variables that we 
        implemented all the way back when we specified our models. Here, we change
        the shared x-variable ```t_shared``` to be the test data rather than the
        training data, and we change the ID variable ```idx_shared``` to be the
        test IDs rather than the train IDs. 
        
        We specify prediction *coords* in order to keep the data format labelled and
        clean and we add the predictions to our idata object. For more in depth
        on *coords*, *dims* and prediction in pyMC3 check out the [example doc](https://docs.pymc.io/notebooks/multilevel_modeling.html)
        and the guide by [oriol](https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/xarray/2020/09/22/pymc3-arviz.html).
        
        '''
        
    '''
    
    # HDI prediction interval 
    
    The HDI prediction intervals look reasonable (i.e. ~80% and ~95% of the actual
    test data-points are inside the 80- and 95% HDI prediction intervals). We note again that it is
    crucial that our uncertainty widens for later time-points. 
    
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
    That concludes our example Bayesian analysis workflow in python (pyMC3) and R (brms). 
    Please get in touch if you have suggestions for improvements. Happy coding & modeling! 
    '''
    
elif choice == "References & Inspiration":
    
    '''
    # Bayesian statistics
    
    * Collection of [good Bayesian books](https://docs.pymc.io/learn.html).
    '''
    
    '''
    # python/pyMC3: 
    
    * Collection of [pyMC3 ressources](https://github.com/pymc-devs/resources), including re-coding of "Rethinking Statistics" (McElreath).
    * [Example notebooks](https://docs.pymc.io/nb_examples/index.html) including some more advanced analyses.
    '''
    
    '''
    # R/brms: 
    
    * Statistical Rethinking (McElreath) recoded in brms by [Solomon Kurz](https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/). 
    * List of [blogposts on brms](https://paul-buerkner.github.io/blog/brms-blogposts/) including some advanced analyses. 
    '''
