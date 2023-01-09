#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,roc_curve,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


# In[133]:


clients_data=pd.read_csv('claimants.csv')
clients_data


# In[134]:


clients_data.head(10)


# In[135]:


clients_data.describe()


# In[136]:


clients_data.isna().sum()


# In[137]:


clients_data.dropna()


# In[138]:


clients_data.dtypes


# In[139]:


del clients_data['CASENUM']


# In[140]:


clients_data.head()


# In[141]:


clients_data.shape


# In[142]:


clients_data.isna().sum()


# In[143]:


clients_data.dropna(axis=0,inplace=True)


# In[144]:


clients_data.shape


# In[145]:


x=clients_data.drop('ATTORNEY',axis=1)
y=clients_data[['ATTORNEY']]


# In[146]:


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.20,random_state=12)


# In[147]:


X_train.shape,Y_train.shape


# In[148]:


X_test.shape,Y_test.shape


# In[ ]:





# In[149]:


logistic_model = LogisticRegression() 
logistic_model.fit(X_train,Y_train)


# In[150]:


logistic_model.coef_


# In[151]:


logistic_model.intercept_


# In[153]:


y_pred_train = logistic_model.predict(X_train)


# In[155]:


print(confusion_matrix(Y_train,y_pred_train))


# In[ ]:





# In[ ]:




