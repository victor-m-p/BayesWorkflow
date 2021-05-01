### NOTE: DOES WORK! 

import streamlit as st
import pandas as pd 
import pickle as pkl 
import altair as alt
import numpy as np 

# setting up matplotlib settings
# Source: https://towardsdatascience.com/making-matplotlib-beautiful-by-default-d0d41e3534fd
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.font_manager as font_manager
import matplotlib.dates as mdates
import pickle as pkl

#%matplotlib inline
from pandas.plotting import scatter_matrix
import seaborn as sns
sns.set(style="whitegrid")
import re
import pyLDAvis


# font
font_dirs = ['/Library/Fonts', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)

plt.rcParams['font.family'] = 'DIN Condensed Bold'

# set matplotlib aesthetics
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

sns.set(rc={
            'axes.axisbelow': False,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'figure.facecolor': 'white',
            'lines.solid_capstyle': 'round',
            'patch.edgecolor': 'w',
            'patch.force_edgecolor': True,
            'text.color': 'dimgrey',
            'xtick.bottom': False,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',
            'xtick.top': False,
            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': False,
            'ytick.right': False,
            'savefig.dpi': 800})

#plt.rcParams["savefig.dpi"] = 'figure'
sns.set_context("notebook", rc={"font.size":12,
                                "axes.titlesize":16,
                                "axes.labelsize":16})


#alt.data_transformers.disable_max_rows()

'''
# Chinese Twitter
'''

expander = st.beta_expander("Introduction to the data")

with expander:
    with open("data/english_clean.pkl", "rb") as f:
        en_df = pkl.load(f)

    en_df['month'] = pd.DatetimeIndex(en_df['created_at']).month
    en_df['date'] = pd.DatetimeIndex(en_df['created_at']).date

    date_freq = pd.DataFrame(en_df.groupby(['date', 'month', "Category"])['created_at'].count()).reset_index()

    date_freq = date_freq.rename(columns = {"date": "Date", "created_at": "Tweet Count"}, inplace = False)

    st.write(alt.Chart(date_freq).mark_line().encode(
        x='Date',
        y='Tweet Count',
        color = 'Category',
        tooltip = ["Date", "Tweet Count", "Category"]
    ).properties(
        width=700,
        height=300
    )
    )

    diplomats = en_df[en_df["Category"] == "Diplomat"]

    date_freq = pd.DataFrame(diplomats.groupby(['date', 'month', "username"])['created_at'].count()).reset_index()

    date_freq = date_freq.rename(columns = {"date": "Date", "created_at": "Tweet Count", "username": "Username"}, inplace = False)

    diplomats = list(set(date_freq["Username"]))

    options_diplo = st.multiselect(
        "What diplomats do you want to inspect?",
        diplomats, diplomats
    )

    date_freq = date_freq[date_freq["Username"].isin(options_diplo)]

    st.write(alt.Chart(date_freq).mark_line().encode(
        x='Date',
        y='Tweet Count',
        color = 'Username',
        tooltip = ["Date", "Tweet Count", "Username"]
    ).properties(
        width=700,
        height=300
    )
    )

expander = st.beta_expander("Overview of the data")

with expander:
    with open("data/2019-11-01_CT.pkl", "rb") as f:
        df = pkl.load(f)

    chart = sns.countplot(
    data = df,
    y="Category",
    palette='inferno'
    )

    chart.set(xlabel='Tweet count', ylabel='Category')
    chart.set_title("Frequency of tweets by Categories")
    st.pyplot(plt)

    plt.clf()

    ## User distribution:
    chart = sns.countplot(
        data=df,
        y='username',
        palette='inferno',
        order = df['username'].value_counts().index)

    chart.set(xlabel='Tweet count', ylabel='Username')
    chart.set_title("Frequency of tweets by Users")
    st.pyplot(plt)

expander = st.beta_expander("Distribution of words")

with expander:
    # Visualize frequent terms
    #from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    #from yellowbrick.text import FreqDistVisualizer, TSNEVisualizer, freqdist
    #from palettable.matplotlib import Inferno_20
    #plt.close()
    # clean text
    #categories = ["Media", "Diplomat"]

    '''
    ### Media 
    '''
    st.image("plots/MediaFreqDist.png")
    #df_filt = en_df[en_df["Category"] == categories[0]]
    #vect = CountVectorizer(stop_words='english', min_df=10, ngram_range=(1,2))
    #docs = vect.fit_transform(df_filt['text_clean'].dropna())
    #features = vect.get_feature_names()

    #freqdist(features, docs, orient = "h", n = 30, color='#371A5C', show = True)
    #st.pyplot(plt)
    #plt.savefig('plots/MediaFreqDist.png', bbox_inches='tight')

    #plt.close("all")

    '''
    ### Diplomat
    '''
    st.image("plots/DiploFreqDist.png")
    #df_filt = en_df[en_df["Category"] == categories[1]]
    #vect = CountVectorizer(stop_words='english', min_df=10, ngram_range=(1,2))
    #docs = vect.fit_transform(df_filt['text_clean'].dropna())
    #features = vect.get_feature_names()

    #freqdist(features, docs, orient = "h", n = 30, color='#371A5C', show = True)
    #st.pyplot(plt)
    #plt.savefig('plots/DiploFreqDist.png', bbox_inches='tight')

'''
## Hashtag plots
'''
with open("data/Media_english_hashtags.pkl", "rb") as f:
    media = pkl.load(f)

with open("data/Diplomat_english_hashtags.pkl", "rb") as f:
    diplo = pkl.load(f)

media["Category"] = "Media"
diplo["Category"] = "Diplomat"

df = pd.concat([media, diplo])

df['just_date'] = df['created_at'].dt.date
#df = df.rename(columns={"Unnamed: 0": "id", "hashtags_bytwitter": "hashtag"})
per_date = df[['just_date', 'id', 'hashtag', "Category"]].groupby(['just_date', 'hashtag', "Category"]).agg(['count']).reset_index()
per_date["hashtag_per_date"] = per_date["id"]["count"]
per_date = per_date[["just_date", "hashtag", "hashtag_per_date", "Category"]]

per_date = per_date.rename(columns={"hashtag": "Hashtag"})

#per_date = per_date.set_index("just_date")

per_date["just_date"] = pd.to_datetime(per_date["just_date"])

### STATIC PLOTS

expander = st.beta_expander("Static Plots of Hashtags")

with expander:

    # FREQ:
    df = per_date
    freq_data = df.groupby(["Category",'Hashtag']).sum().reset_index().rename(columns={"hashtag_per_date": "count"})

    freq_data = freq_data.sort_values(by=['count'], ascending=False).reset_index(drop=False)

    freq_media = freq_data[freq_data["Category"] == "Media"].head(20)
    freq_diplomat = freq_data[freq_data["Category"] == "Diplomat"].head(20)

    freq_comb = pd.concat([freq_diplomat, freq_media])

    #freq_data = freq_data[freq_data["count"] > 500]
    nr_hash = len(freq_comb["Hashtag"].unique())

    sns.set()
    palette = sns.color_palette("inferno", nr_hash)

    g = sns.FacetGrid(data = freq_comb, row = "Category",
                            palette = palette,
                            sharey=False,
                            sharex= False,
                            height = 4,
                            aspect = 3,
                            )
    g.map(sns.barplot, "count", "Hashtag", palette = palette)
    g.add_legend()
    st.pyplot(g)

    ### FREQ OVER TIME:
    freq_time = df.groupby(["Category",'Hashtag', "just_date"]).sum().reset_index().rename(columns={"hashtag_per_date": "count"})

    freq_time_media = freq_time[(freq_time["Category"] == "Media") & (freq_time["Hashtag"].isin(list(freq_media.head(5)["Hashtag"])))]
    freq_time_diplomat = freq_time[(freq_time["Category"] == "Diplomat") & (freq_time["Hashtag"].isin(list(freq_diplomat.head(5)["Hashtag"])))]
    freq_time_comb = pd.concat([freq_time_diplomat, freq_time_media])

    nr_hash = len(freq_time_comb["Hashtag"].unique())
    palette = sns.color_palette("inferno", nr_hash)

    g = sns.FacetGrid(data = freq_time_comb, row = "Category", hue = "Hashtag",
                            palette = palette,
                            sharey=False,
                            sharex= True,
                            height = 4,
                            aspect = 3,
                            )
    g.map(sns.lineplot, "just_date", "count")
    g.add_legend()
    st.pyplot(g)

### DYNAMIC PLOTS

hashtags = list(set(per_date["Hashtag"]))
options = st.multiselect(
        "What hashtags do you want to compare?",
        hashtags, ["#china", "#coronavirus", "#us"]
    )


expander = st.beta_expander("Hashtag Frequency Plots")

with expander:


    ### PLOTTING: 

    import seaborn as sns; sns.set()
    import matplotlib.pyplot as plt
    import pyplot_themes as themes

    df = per_date
    df = df.loc[df['Hashtag'].isin(options)]
    df = df.rename(columns = {"just_date":"Date", "hashtag_per_date":"Frequency"})

    df = pd.DataFrame({
        "Date": df["Date"].values,
        "Frequency": df["Frequency"].values,
        "Hashtag": df["Hashtag"].values,
        "Category": df["Category"].values
    })

    st.write(alt.Chart(df).mark_line().encode(
        x='Date',
        y='Frequency',
        color = 'Hashtag',
        tooltip = ["Date", "Frequency", "Hashtag", "Category"]
    ).properties(
        width=600,
        height=300
    ).facet(
        row='Category',
    ).resolve_scale(y='independent')
    )

expander = st.beta_expander("Centrality plots")

with expander:
    with open("data/networks/bin_dict.pkl", "rb") as f:
        concat_bins = pkl.load(f)

    media_bins = concat_bins["Media"]
    diplo_bins = concat_bins["Diplomat"]

    plot_names = list(media_bins.keys())

    def get_centrals(concat_bins, name):

        ## Centrality over time plot

        centrals = [(i,j) for j in plot_names for i in concat_bins[j]["centrality"] if i[0] in options]

        central_df = pd.DataFrame({
            "Date": [i[1] for i in centrals],
            "Hashtag": [i[0][0] for i in centrals],
            "Centrality": [i[0][1] for i in centrals]
        })
        central_df["Category"] = name
        return central_df

    indexer = [("Media", media_bins), ("Diplomat", diplo_bins)]

    central_df = pd.concat([get_centrals(i[1], i[0]) for i in indexer])

    st.write(alt.Chart(central_df).mark_line().encode(
        x='Date',
        y='Centrality',
        color = 'Hashtag',
        tooltip = ["Date", "Centrality", "Hashtag"]
    ).properties(
        width=600,
        height=300
    ).facet(
        row='Category',
    ).resolve_scale(y='independent')
    )

### TOPIC MODELLING:

expander = st.beta_expander("Topic Modelling")

with expander:

    with open("plots/Media_pyLDAvis_plot.pkl", "rb") as f:
        media_lda = pkl.load(f)

    with open("plots/Diplomat_pyLDAvis_plot.pkl", "rb") as f:
        diplo_lda = pkl.load(f)
    
    diplo_string = pyLDAvis.prepared_data_to_html(diplo_lda)
    media_string = pyLDAvis.prepared_data_to_html(media_lda)
    '''
    ## Diplomat
    '''

    from streamlit import components

    components.v1.html(diplo_string, width=1300, height=800)
    #st.markdown(diplo_string, unsafe_allow_html=True)

    '''
    ## Media
    '''
    #st.markdown(media_string, unsafe_allow_html=True)
    components.v1.html(media_string, width=1300, height=800)
    #st.markdown(pyLDAvis.display(media_lda), unsafe_allow_html=True)