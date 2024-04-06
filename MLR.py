#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("/Users/tugcesandikli/Downloads/advertising.csv")


# In[3]:


data.head()


# In[5]:


x = data[["TV","Radio","Newspaper"]]
y = data["Sales"]


# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.3,random_state=100)


# In[7]:


from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train,y_train)


# In[8]:


print("intercep",mlr.intercept_)
print("coefficients:")
list(zip(x,mlr.coef_))


# In[12]:


y_pred = mlr.predict(x_test)
print("prediction{}".format(y_pred))


# In[14]:


mlr_dif = pd.DataFrame({"gercek deger": y_test,"tahmin edilen": y_pred})
mlr_dif.head()


# In[18]:


from sklearn import metrics

meansqrer= metrics.mean_squared_error(y_test,y_pred)
print("r sqr{:.2f}".format(mlr.score(x,y)*100))
print("msqrerr:",meansqrer)

