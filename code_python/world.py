# importing stuff
import streamlit as st
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# side-bar would be good. 
# really need search. 
# really need hover. 
# world map is really cool. 

'''
# "Hello World"! 
'''

# image (as background??)
from PIL import Image
import os 
st.image(Image.open(os.path.join("world-img.jpg")))

'''

This is a playground for exploring the the world. Here, you are the master.

Questions such as: 

* "Can money buy happiness?" 

* "Is Denmark really the happiest country in the world?"

* "What has love got to do with it?" 

Can be answered here! Well... possibly not the last one... 

'''


'''
# Just a quick question before you continue.. 
'''

# could use this for something..
knowledge = st.radio("What is your statistics knowledge?", ("Layman", "I have seen a scatterplot before", "Expert"))

# greet them
if knowledge == "Layman": 
    st.text('We will take a gentle ride!')
elif knowledge == "I have seen a scatterplot before":
    st.text('You are prepared then...')
else: 
    st.text('Oh, an expert. Interesting.')

## our data set 
df = '2015.csv'


# read data & quick preprocessing. 
#@st.cache(persist = True) #decorator
def load_data(dataset):

    # read the data set 
    df = pd.read_csv(dataset)

    # drop some columns 
    df = df.drop(['Happiness Rank', 'Standard Error', 'Family', 'Dystopia Residual'], axis = 1)

    # rename some columns 
    df.rename(columns={'Economy (GDP per Capita)': 'GDP per capita', 
                     'Health (Life Expectancy)': 'Life expectancy',
                     'Trust (Government Corruption)': 'Trust'}, inplace=True)

    df = df.sort_values(by = "Country", axis = 0, ascending = True)

    return(df)
    

# gapminder stuff instead 
#g_15 = pd.read_csv("2015.csv")

#g_15 = g_15.drop(['Happiness Rank', 'Standard Error', 'Family', 'Dystopia Residual'], axis = 1)

# rename some columns
#g_15.rename(columns={'Economy (GDP per Capita)': 'GDP per capita', 
#                     'Health (Life Expectancy)': 'Life expectancy',
#                     'Trust (Government Corruption)': 'Trust'}, inplace=True)

'''
## 1. Our data. 
Take a quick glance at the variables. 
Then move on.
'''

# show this properly
if st.checkbox("Preview Dataset"):
    data = load_data(df)
    if st.button('Head'): 
        st.write(data.head())
    elif st.button('Tail'): 
        st.write(data.tail())

# very simple function
# maybe regplot?
# implement hover functionality. 
def plot_scatter(df, x, y, size = None, hue = None): 
    
    p = sns.scatterplot(x = x, y = y, 
                    size = size, hue = hue, 
                    sizes = (5, 150),
                    alpha = 0.5,
                    data = df)

    plt.legend(loc = 'upper left', fontsize = 'xx-small')

    fig = p.get_figure()
    
    return(fig)
    #st.pyplot(fig)

# helper function
# create column:
def choose_country(df, country): 
    df["Choice"] = df['Country'].apply(lambda x: country if x == country else "World")
    return(df)

# st.slider("name", min, max)
if knowledge == "Layman": 

    '''
    ## 2. Exploring associations 
    Since you chose "Layman" I will let you play with
    a simple scatterplot.
    In this first plot you can explore associations between
    variables in the data (2015). Each dot you are seeing 
    corresponds to one country. I have sat the x-axis to 
    represent GDP per capita and the y-axis to represent
    life expectancy, but feel free to change these values. 
    If you are interested in a specific country, you can 
    select that as well. If you feel frisky, you can go back
    and select "Expert". This will give you more power
    (but also more responsibility). 
    '''

    g_15 = load_data(df)
    country_cols = np.array(g_15['Country'])
    nonetype = np.array(['None'])
    all_cols = np.append(nonetype, country_cols)

    # selecting stuff
    x = st.selectbox('Select X variable', g_15.columns, 3)
    y = st.selectbox('Select Y variable', g_15.columns, 4)
    country = st.selectbox('Highlight a country', all_cols, 0)

    # whether or not country was selected: 
    if country == "None": 
        fig1 = plot_scatter(g_15, x = x, y = y)
    else: 
        g_15 = choose_country(g_15, country)
        fig1 = plot_scatter(g_15, x = x, y = y, hue = "Choice")
    
    st.pyplot(fig1)

else: 
    '''
    ## 2. Exploring associations 
    Since you are experienced, I will 
    let you plot in *FOUR* dimensions. 
    For instance, you can add the size of the
    dots on "Happiness Rank". Additionally, you could
    also base the color (hue) of the dots on things
    like "Region" to explore whether countries in 
    the same parts of the world show similar patterns. 
    It is of course up to you! 
    '''

    g_15 = load_data(df)
    col_names = np.array(g_15.columns)
    nonetype = np.array(['None'])
    all_cols = np.append(nonetype, col_names)

    # selecting stuff
    x = st.selectbox('Select X variable', g_15.columns, 3)
    y = st.selectbox('Select Y variable', g_15.columns, 4)
    size = st.selectbox('Select SIZE variable', all_cols, 0)
    hue = st.selectbox('Select COLOR to group by', all_cols, 0)

    # different kinds of plots: 
    # make this smarter...
    if size == "None" and hue == "None": 
        fig2 = plot_scatter(g_15, x = x, y = y)
    elif size == "None" and hue != "None": 
        fig2 = plot_scatter(g_15, x = x, y = y, hue = hue)
    elif size != "None" and hue == "None": 
        fig2 = plot_scatter(g_15, x = x, y = y, size = size)
    else: 
        fig2 = plot_scatter(g_15, x = x, y = y, size = size, hue = hue)

    st.pyplot(fig2)


# new function
def plot_lm(df, x, y, hue = None): 
    
    p = sns.lmplot(x = x, y = y, hue = hue,
                    data = df)

    #plt.legend(loc = 'upper left', fontsize = 'xx-small')

    #fig = p.get_figure()
    
    return(p)

'''
## 3. Regression. 
Now we turn to regressions. 
Can you find surprising relationships in the data?
Perhaps you already have some ideas from the earlier
plotting that you did. 
You can group the plot by a grouping variable

'''

g_15 = load_data(df)
col_names = np.array(g_15.columns)
nonetype = np.array(['None'])
all_cols = np.append(nonetype, col_names) # should only be numerics. 

# selecting stuff
x = st.selectbox('Select X variable (again)', g_15.columns, 3)
y = st.selectbox('Select Y variable (again)', g_15.columns, 4)
hue_col = st.selectbox('Select grouping variable', all_cols, 0)

# what is wrong with qcut..?
if hue_col == "None": 
    fig3 = plot_lm(df = g_15, x = x, y = y)
elif hue_col != "None": 
    bin_labels = ['low', 'low medium', 'high medium', 'high']
    g_15[hue_col] = pd.qcut(g_15[hue_col], q = [0, 0.25, 0.5, 0.75, 1], labels = bin_labels)
    fig3 = plot_lm(df = g_15, x = x, y = y, hue = hue_col)

st.pyplot(fig3)

'''
## 4. I forgot about time...
Now you have hopefully explored our little data set. 
What you have seen so far is data from 2015.
I will now allow you to tweak one last thing: 
Time. (Come back, this data is crazy). 
'''

