#!/usr/bin/env python
# coding: utf-8

# # Simple linear Regression
# 
# building linear regression models to predict sales using an appropriate predict variable

# # Reading and Understanding data

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# In[3]:


adv=pd.read_csv("advertising.csv")
adv.head()


# In[4]:


adv.shape


# In[5]:


adv.info()


# In[6]:


adv.describe()


# ### visualising the data

# In[7]:


sns.pairplot(x_vars=["TV","Radio","Newspaper"],y_vars="Sales",data=adv)


# In[8]:


sns.heatmap(adv.corr(),annot=True)


# ### Generic Steps in model building using `statsmodels`
# 
# We first assign the feature variable, `TV`, in this case, to the variable `X` and the response variable, `Sales`, to the variable `y`.

# In[9]:


X=adv['TV']
y=adv['Sales']


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.70,random_state=100)


# In[11]:


X_train_sm=sm.add_constant(X_train)
X_train_sm.head()


# In[12]:


lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
lr_model.params


# In[13]:


lr_model.summary()


# In[14]:


y_train_pred=lr_model.predict(X_train_sm)


# In[15]:


plt.scatter(X_train,y_train)
plt.plot(X_train,y_train_pred,'r')
plt.show()


# ## Residual analysis 
# To validate assumptions of the model, and hence the reliability for inference

# In[16]:


res=y_train-y_train_pred


# In[17]:


plt.figure()
sns.distplot(res)
plt.title("Residual Plot")
plt.show()


# ##  Predictions on the Test Set

# In[18]:


X_test_sm=sm.add_constant(X_test)

y_test_pred=lr_model.predict(X_test_sm)


# In[19]:


r2=r2_score(y_true=y_test,y_pred=y_test_pred)
r2


# In[20]:


plt.scatter(X_test,y_test)
plt.plot(X_test,y_test_pred,'r')
plt.show()


# ### Linear Regression using `linear_model` in `sklearn`

# In[21]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=100)


# In[23]:


X_train_lm=X_train.values.reshape(-1,1)
X_test_lm=X_test.values.reshape(-1,1)


# In[24]:


lm=LinearRegression()
lm.fit(X_train_lm,y_train)


# In[25]:


print(lm.coef_)
print(lm.intercept_)


# In[26]:


y_train_pred=lm.predict(X_train_lm)
y_test_pred=lm.predict(X_test_lm)


# In[28]:


print(r2_score(y_true=y_train,y_pred=y_train_pred))
print(r2_score(y_true=y_test,y_pred=y_test_pred))

