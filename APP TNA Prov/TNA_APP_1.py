#!/usr/bin/env python
# coding: utf-8

# # Transferencias no Automáticas a Provincias - APP

# In[74]:


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

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("max_rows", None)

#APP
import streamlit as st


# ### ReadMe

# In the following Notebook we are going to create the app that allow us to create a dashboard that facilitate reading the dataset of the transfers from the federal government to the provincies.

# In[77]:


st.title('Creating Web Apps with JN and Streamlit')

st.subheader('Valentin Mongi')

st.title('The Project') 

st.markdown('>This project uses the daily information of all the expenditures of the central governemnt. We are going to analyse the discretional transfers from the federal governemnt to the provincies in the period 2010 - 2020.')


# ## Import DATASET

# In[82]:


st.code("""
@st.cache
def get_data():
    url = 'C:\Users\valen\Google Drive\Machine Learning\ML - GitHub\APP TNA Prov\Dataset\final_vf.csv'
    return pd.read_csv(url, parse_dates=['Fecha'])
""", language="python")


# In[83]:


data = pd.read_csv(r'Dataset\final_vf.csv', parse_dates=['Fecha'])


# In[84]:


data.head()


# ### Create the DashBoard

# In[85]:


pob =  pd.DataFrame(data, columns=['Fecha', 'Provincia', 'tna(nominal)_pc'])
pob['Fecha'] = pd.PeriodIndex(pob['Fecha'], freq='A')

pob = pob.set_index(['Fecha'])


# In[86]:


pob_1 = pob['2019'].groupby(['Provincia']).sum().reset_index()
pob_2 = pob_1.sort_values(by='tna(nominal)_pc', ascending=False)


# In[87]:


pob_2.head()


# In[88]:

plt.figure(figsize=(15,6))
sns.barplot(x='Provincia', y='tna(nominal)_pc', data=pob_2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.despine()


st.bar_chart(pob_2)


# In[89]:





# In[ ]:




