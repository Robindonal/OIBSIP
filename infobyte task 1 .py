#!/usr/bin/env python
# coding: utf-8

# # ROBIN DONAL

# # TASK - 1

# # IRIS FLOWER CLASSIFICATION 

# In[1]:


import pandas as pd


# # Reading The DataSet 

# In[2]:


df=pd.read_csv("Iris.csv")


# In[3]:


df


# # Load the Iris dataset

# In[4]:


from sklearn.datasets import load_iris


# In[5]:


iris = load_iris()


# In[6]:


X, y = iris.data, iris.target


# # Split the data into training and testing sets

# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Choose a machine learning algorithm

# In[9]:


from sklearn.linear_model import LogisticRegression


# In[10]:


model = LogisticRegression()


# # Train The Model 

# In[11]:


model.fit(X_train, y_train)


# # Evaluate the model

# In[12]:


from sklearn.metrics import accuracy_score


# In[13]:


y_pred = model.predict(X_test)


# In[14]:


accuracy = accuracy_score(y_test, y_pred)


# In[15]:


print(f"Accuracy: {accuracy}")

