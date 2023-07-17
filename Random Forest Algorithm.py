#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load Iris dataset
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
#Set random seed
np.random.seed()


# In[2]:


iris =load_iris()
df= pd.DataFrame(iris.data, columns=iris.feature_names)


# In[6]:


df['species'] =pd.Categorical.from_codes(iris.target,iris.target_names)


# In[7]:


df


# In[8]:


df['is_train']=np.random.uniform(0,1, len(df)) <= .75


# In[9]:


df


# In[13]:


# Create dataframes with test rows and training rows
train, test = df[df['is_train']==True],df[df['is_train']==False]
print('Number of observations in the training data:',len(train))
print('Number of observations in the test data:',len(test))


# In[14]:


#Create a list of columns names
features =df.columns[:4]


# In[15]:


features


# In[17]:


#converting each species name
y =pd.factorize(train['species'])[0]


# In[18]:


y


# In[20]:


#creating a random forest Classifier
clf =RandomForestClassifier(n_jobs=2, random_state =0)
#Training the classifier
clf.fit(train[features],y)


# In[22]:


clf.predict(test[features])


# In[23]:


test[features]


# In[24]:


clf.predict_proba(test[features])[0:10]


# In[25]:


#mapping names for the plants for each predicated plant class
preds =iris.target_names[clf.predict(test[features])]
preds[0:25]


# In[26]:


test['species']


# In[27]:


#creating confusion matrix
pd.crosstab(test['species'], preds, rownames=['Actual Species'],
colnames=['Predicted Species'])


# In[30]:


preds =iris.target_names[clf.predict([[5.0,3.6,1.4,2.0],[5.0,3.6,1.4,2.0]])]
preds


# In[ ]:




