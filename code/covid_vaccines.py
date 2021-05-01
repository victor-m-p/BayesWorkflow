import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="Covid Vaccines",
    page_icon="bayes_bois.png",
    layout="wide",
    initial_sidebar_state="expanded",
    )

st.image("bayes_bois.png", width = 100)

"""
# Covid Vaccines - how are things going? 

Let's visualize how vaccinations are going over time! 

"""
### Expander for the data 
expander = st.beta_expander("The Data")
expander.write(f"The data was downloaded from Kaggle. However, several preproccessing steps were taken. The countries differed in terms of how many days they had recorded data. This was padded with NAs. Furthermore, the $pd.DateTime()$ function was used extensively. You can find the data here: https://www.kaggle.com/gpreda/covid-world-vaccination-progress")

# Preprocessing

df = pd.read_csv("country_vaccinations.csv")

df_pivot = df.pivot(columns = "country", values = "daily_vaccinations_per_million")

countries = list(df_pivot.columns)

df["date"] = pd.to_datetime(df["date"])

date_ranging = pd.date_range(min(df["date"]), max(df["date"]))

## Make padding for data-frames

list_of_dfs = []

for i in countries:
    df_new = df[df["country"] == i]
    df_new = df_new.set_index("date").reindex(date_ranging).fillna(np.nan).rename_axis("date").reset_index()
    df_new["country"] = i
    list_of_dfs.append(df_new)

df = pd.concat(list_of_dfs)

## Rename columns

new_names = {'total_vaccinations':"Total Vaccinations", 'people_vaccinated': "People Vaccinated", 'people_fully_vaccinated': "People Fully Vaccinated", 'daily_vaccinations_raw': "Raw Daily Vaccinations", 'daily_vaccinations': "Daily Vaccinations", 'total_vaccinations_per_hundred':"Total Vaccinations Pr. Hundred", 'people_vaccinated_per_hundred':"People Vaccinated Pr. Hundred", 'people_fully_vaccinated_per_hundred': "People Fully Vaccinated Pr. Hundred", 'daily_vaccinations_per_million': "Daily Vaccinations Pr. Million"}

df = df.rename(columns = new_names)

### Checkbox for showing the dataframe
with expander:
    if st.checkbox('Show the dataframe'):
        df

expander = st.beta_expander("The World Map")
expander.write("Want to get a better geographical understanding of the countries involved in this data-set? Press the button below to get a better sense of where those countries are in the world.")

with expander:
    if st.checkbox("Show a world map"):
        st.map()

'''
## Guide to the interactive plot
'''

## Multi selecter for the countries

options = st.sidebar.multiselect(
    "What countries do you want to compare?",
    countries, None
)

## Columns to use for y-columns

interesting_columns = list(new_names.values())

### Selecter for the y-column

y_col = st.sidebar.selectbox('What variable would you like to compare between countries?', interesting_columns)

### Selecter for interpolating values

interpol = st.sidebar.selectbox('Do you want to interpolate values of NA?', ["Yes", "No"])

'''
This website is for plotting the development of vaccines for COVID-19 for each country over the recent month. 
Select different values on the left to explore the data. When you select a country or more, your specificed plot will show below. Below these plots are explanations for each selection you can make.
'''

# Main plotting

if options:
    dictionary_comp = {option: df[df["country"] == option][y_col].values for option in options}
    dictionary_comp["date"] = list(date_ranging)
    df_subset = pd.DataFrame(dictionary_comp)

    df_subset = df_subset.rename(columns={'date':'index'}).set_index('index')


    if interpol == "Yes":
        df_subset = df_subset.interpolate(method = "time")
    
    ## CURRENTLY NOT SHOWING y-label. This can be fixed here: https://github.com/streamlit/streamlit/issues/1129

    st.line_chart(df_subset)

    expander = st.beta_expander("Area Chart")
    expander.write(f"The plot belows shows the area under the curve for the variable {y_col}. When multiple countries are chosen, the countries will be stacked on top of each other. As such, this plot is best used to get better intuitions about how countries differ in terms of volume.")

    st.area_chart(df_subset)

    ### Expander for countries

    '''
    ## Information regarding interactive variables
    '''

    expander = st.beta_expander("Countries")
    expander.write(f"This variable is used to select the countries you want to compare. Currently you have selected: {options}")

    ### Expander for the y-axis

    expander = st.beta_expander("The Y-axis")
    expander.write(f"The second variable determines what you want to compare the countries on. You have currently selected {y_col}.")

    ### Expander for interpolation

    expander = st.beta_expander("Interpolation")
    expander.write(f"The third variable is used to select whether to interpolate values when there are NAs in the data. This is a consequence of there being a disparity between how much data each country has. Some have just started later than other. The method used is $pd.interpolate(method = time)$. To the question of whether or not to interpolate, you have currently answered '{interpol}!'")

    non_interpolation_df = df[df["country"] == "Argentina"].set_index("date")
    interpolation_df = non_interpolation_df.interpolate(method = "time")

    dictionary_int = {
        "Original Data": non_interpolation_df["Total Vaccinations Pr. Hundred"].values, 
        "Interpolated Data": interpolation_df["Total Vaccinations Pr. Hundred"].values
        }

    int_df = pd.DataFrame(dictionary_int)

    expander.area_chart(int_df)
    

for i in range(1000):
    inputting = input("hey")
    if inputting == "break":
        break