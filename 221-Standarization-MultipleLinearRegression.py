#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression


# In[4]:


data = pd.read_csv('1.02. Multiple linear regression.csv')
data.head()


# In[5]:


data.describe()


# In[6]:


x = data[['SAT','Rand 1,2,3']]
y = data['GPA']


# ## Standarization

# In[9]:


from sklearn.preprocessing import StandardScaler


# In[11]:


scaler = StandardScaler()


# In[12]:


scaler.fit(x)


# In[13]:


x_scale = scaler.transform(x)


# In[14]:


x_scale


# In[15]:


reg = LinearRegression()
reg.fit(x_scale,y)


# In[16]:


reg.coef_


# In[18]:


reg.intercept_


# In[19]:


reg_summary = pd.DataFrame([['Intercept'],['SAT'],['Rand 1,2,3']], columns=['Features'])
reg_summary ['Weight'] = reg.intercept_, reg.coef_[0], reg.coef_[1]


# In[20]:


reg_summary


# In[21]:


#biggr the number, bigger the impact
#weight is known as coeficients
# intercept is known as bias - coenfficient with standarization


# In[22]:


#same as above
reg_summary = pd.DataFrame([['Bias'],['SAT'],['Rand 1,2,3']], columns=['Features'])
reg_summary ['Weight'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
reg_summary


# In[23]:


new_data = pd.DataFrame(data=[[1700,2],[1800,1]], columns=['SAT','Rand 1,2,3'])
new_data


# In[24]:


reg.predict(new_data)


# In[26]:


## the result above, doesn't make sense at all, it is because we need to standarize all the related information too


# In[28]:


new_data_scaled = scaler.transform(new_data)
new_data_scaled


# In[29]:


reg.predict(new_data_scaled)


# In[ ]:


## Now the results without the random var


# In[32]:


reg_simple = LinearRegression()
x_simple_matrix = x_scale[:,0].reshape(-1,1)
reg_simple.fit(x_simple_matrix,y)


# In[35]:


reg_simple.predict(new_data_scaled[:,0].reshape(-1,1))


# In[ ]:


##This show us that the random var is not relevant


# In[ ]:





# In[ ]:




