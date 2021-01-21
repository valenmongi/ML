#!/usr/bin/env python
# coding: utf-8

# # Transferencias no Automáticas a Provincias - APP

# In[1]:


import scipy
from scipy import stats

#librerías necesarias:

# Data handling
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Visualization
import seaborn as sns
#import plotly.express as px
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

import nbconvert

#APP
import streamlit as st
import altair as alt
from altair import Chart, X, Y, Axis, SortField, OpacityValue


from PIL import Image


# ### ReadMe

# In the following Notebook we are going to create the app that allow us to create a dashboard that facilitate reading the dataset of the transfers from the federal government to the provincies.

# In[23]:


st.title('Creating Web Apps with JN and Streamlit')

st.markdown('## Valentin Mongi')

st.title('The Project') 

st.markdown('>This project uses the daily information of all the expenditures of the central governemnt. We are going to analyse the discretional transfers from the federal governemnt to the provincies in the period 2010 - 2020.')


# In[24]:


image = Image.open('flag.jpg')
st.image(image, use_column_width=True)


# ## Import DATASET

# In[25]:


data = pd.read_csv(r'Dataset\final_vf.csv', parse_dates=['Fecha'])


# ### Create the DashBoard

# In[26]:


pob =  pd.DataFrame(data, columns=['Fecha', 'Provincia', 'tna(nominal)_pc'])
pob['Fecha'] = pd.PeriodIndex(pob['Fecha'], freq='A')
pob = pob.rename(columns={'tna(nominal)_pc':'TNA_percapita'})


# In[27]:


df = pob.groupby(['Fecha', 'Provincia']).sum().reset_index()


# In[28]:


df['Fecha'] = df['Fecha'].astype(str)


# In[29]:


df['Fecha'] = df['Fecha'].astype(int)


# Groupby Years

# In[30]:


pob_1 = pob[pob['Fecha'] == '2019'].groupby(['Provincia']).sum().reset_index()
pob_2 = pob_1.sort_values(by='TNA_percapita', ascending=False)


# ## First, create a historical graph

# We are going to create a evolutive graph that shows the ranking of transfers per capita, by province and year.

# In[31]:


st.markdown('### Ranking of Transfer per capita by Provinces')
st.markdown('*We are going to build a evolutive graph that shows the ranking of the provinces who were more benefitted with the transfers from the federal governmet. The variable included is the **nominal transfers per capita**.*')


# In[33]:


select_year = alt.selection_single(name='select', fields=['Fecha'], init={'Fecha': 2010},
        bind = alt.binding_range(min=2010, max=2020, step=1))

base = alt.Chart(df, title="Ranking as of year").mark_bar().encode(
    x=alt.X('TNA_percapita', title='TNA.percapita'), 
    y=alt.Y('Provincia', sort='-x'), 
    color=alt.Color('Provincia'),
    tooltip = [alt.Tooltip('Provincia'),
               alt.Tooltip('TNA_percapita')]).properties(width=650, height=400).add_selection(select_year).transform_filter(select_year)


# In[34]:


st.altair_chart(base)
st.balloons() 


# ## Analyzing the data per provinces

# First we are going to create a **INTERACTIVE** multi selector that allows to filter by provinces and years.

# **Streamlit MultiSelector by provinces and years.**

# In[14]:


st.markdown('### Selector')
st.markdown('In this section we are goint to consider the transfers to provincies over the population of each one. You are going to be able to select the year and provinces of your interest to create your own interactive graph.')


# In[15]:


# Year selector

year = st.slider("Choose year of interest: ", min_value=2010,   
                       max_value=2020,step=1)


# In[16]:


p = pd.Series(pob['Provincia'].unique()).sort_values()


# In[17]:


# Provinces selector


provinces = st.multiselect("Which Provinces you want to consider?", 
                        p)


# In[18]:


# st.write(data[data.label == desired_label])


# ## Graphs

# In[19]:


plt.figure(figsize=(15,6))
sns.barplot(x='Provincia', y='TNA_percapita', data=pob_2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.despine()
st.pyplot()


# In[20]:


st.set_option('deprecation.showPyplotGlobalUse', False)


# In[21]:


get_ipython().system('jupyter nbconvert --to script TNA_APP.ipynb')


# In[ ]:





# In[ ]:




