#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# In[8]:


data = pd.read_csv('Random_Features.csv')
df_shuffled = data.sample(frac=1, random_state=42) #Shuffle the df to avoid overfitting
df_shuffled.reset_index(drop=True, inplace=True)


# In[9]:


# Assuming the last column is the target variable and the rest are features
X = df_shuffled.iloc[:, 2:-1]
y = df_shuffled.iloc[:, -1]


# In[10]:


seed_value = 42
model = RandomForestClassifier(random_state=seed_value)

step = 1  # The number of features to remove at each iteration
n_features_to_select = 20  # If None, half of the features are selected by default

# RFE with step and n_features_to_select
rfe = RFE(model, n_features_to_select=n_features_to_select, step=step)
fit = rfe.fit(X, y)

# Display top features along with their importance scores
feature_importances = pd.Series(fit.estimator_.feature_importances_)
top_features = feature_importances.nlargest(len(feature_importances))

print("Top Features and Their Importance Scores:")
for i in top_features.index:
    print(f"Feature {i + 1}: {X.columns[i]}, Importance Score: {top_features[i]}")

# Get top features in a list with ' ' brackets
top_features_list = list(X.columns[fit.support_])
print("Top Features in List:", top_features_list)

selected_features = X.iloc[:, fit.support_]
model.fit(selected_features, y)


# In[ ]:




