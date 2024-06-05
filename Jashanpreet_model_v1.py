#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ucimlrepo')


# In[2]:


from ucimlrepo import fetch_ucirepo


# In[3]:


# Fetching the Dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 


# In[4]:


import pandas as pd


# In[5]:


# Data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 

# Metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 

# Variable information 
print(breast_cancer_wisconsin_diagnostic.variables)


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[7]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{matrix}")


# In[ ]:




