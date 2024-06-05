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


# In[8]:


from sklearn.tree import DecisionTreeClassifier

# Train a Decision Tree model
model_v2 = DecisionTreeClassifier()
model_v2.fit(X_train, y_train)

# Make predictions
y_pred_v2 = model_v2.predict(X_test)

# Evaluate the model
accuracy_v2 = accuracy_score(y_test, y_pred_v2)
report_v2 = classification_report(y_test, y_pred_v2)
matrix_v2 = confusion_matrix(y_test, y_pred_v2)

print(f"Accuracy: {accuracy_v2}")
print(f"Classification Report:\n{report_v2}")
print(f"Confusion Matrix:\n{matrix_v2}")


# In[ ]:




