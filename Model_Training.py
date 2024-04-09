#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Developers: Kunjulakshmi R, Dedeepya Mullapudi, Avik Sengupta


# In[1]:


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import random
from scipy.stats import uniform
# random search
from scipy.stats import loguniform
from scipy.stats import randint
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle as pkl
from xgboost import XGBClassifier
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline


# In[2]:


df = pd.read_csv('Random_Features.csv')
# get the locations
x = df.iloc[:, 2:-1]
y = df.iloc[:, -1]

# split the datase
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)


# In[3]:


df


# In[4]:


x


# In[5]:


#Standard Scaling
sc= StandardScaler()
x_train_sc= pd.DataFrame(sc.fit_transform(x_train), columns= x_train.columns)
x_test_sc= pd.DataFrame(sc.fit_transform(x_test), columns= x_test.columns)


# In[6]:


x_train_sc


# In[7]:


# Create an instance of the SVM classifier
svm_model = SVC(kernel='linear', random_state=42) # Linear Kernel

# Train the model using the training sets
svm_model.fit(x_train_sc, y_train)

# Create a KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and get cross-validated predictions
y_pred_train_svm = cross_val_predict(svm_model, x_train_sc, y_train, cv=kf)


# In[8]:


y_actual_train_svm = y_train

tn1,fp1,fn1,tp1 = confusion_matrix(y_actual_train_svm, y_pred_train_svm).ravel()
print('SVC')
print("True Positives ::",tp1)
print("True Negatives ::",tn1)
print("False Positives ::",fp1)
print("False Negatives ::",fn1)
Accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
print("Accuracy {:0.4f}".format(Accuracy1))
Specificity1 = tn1/(tn1+fp1)
print("Specificty {:0.4f}".format(Specificity1))
Sensitivity1 = tp1/(tp1+fn1)
print("Sensitivity {:0.4f}".format(Sensitivity1))
Precision1 = tp1/(tp1+fp1)
print("Precision {:0.4f}".format(Precision1))
F1_Score1 = 2*(Precision1*Sensitivity1)/(Precision1+Sensitivity1)
print("F1_Score {:-.4f}".format(F1_Score1))
MCC1 = matthews_corrcoef(y_actual_train_svm, y_pred_train_svm)
print("MCC {:0.4f}".format(MCC1))
ROC1 = metrics.roc_auc_score(y_actual_train_svm, y_pred_train_svm)
print("ROC {:0.4f}".format(ROC1))


# In[9]:


y_pred_test_svm = svm_model.predict(x_test_sc)


# In[10]:


y_actual_test_svm = y_test

tn1,fp1,fn1,tp1 = confusion_matrix(y_actual_test_svm, y_pred_test_svm).ravel()
print('SVC')
print("True Positives ::",tp1)
print("True Negatives ::",tn1)
print("False Positives ::",fp1)
print("False Negatives ::",fn1)
Accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
print("Accuracy {:0.4f}".format(Accuracy1))
Specificity1 = tn1/(tn1+fp1)
print("Specificty {:0.4f}".format(Specificity1))
Sensitivity1 = tp1/(tp1+fn1)
print("Sensitivity {:0.4f}".format(Sensitivity1))
Precision1 = tp1/(tp1+fp1)
print("Precision {:0.4f}".format(Precision1))
F1_Score1 = 2*(Precision1*Sensitivity1)/(Precision1+Sensitivity1)
print("F1_Score {:-.4f}".format(F1_Score1))
MCC1 = matthews_corrcoef(y_actual_test_svm, y_pred_test_svm)
print("MCC {:0.4f}".format(MCC1))
ROC1 = metrics.roc_auc_score(y_actual_test_svm, y_pred_test_svm)
print("ROC {:0.4f}".format(ROC1))


# In[11]:


#create an instance of the RandomForestClassifier
rfc_model = RandomForestClassifier(random_state=42)

#Fitting the model
rfc_model.fit(x_train, y_train)

# Create a KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and get cross-validated predictions
y_pred_train_rfc = cross_val_predict(rfc_model, x_train, y_train, cv=kf)


# In[39]:


def plot_roc_curve(y_actual_train_rfc, y_pred_train_rfc):
    fpr, tpr, thresholds = roc_curve(y_actual_train_rfc, y_pred_train_rfc)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(y_actual_train_rfc, y_pred_train_rfc)
print(f'model 2 AUC score: {roc_auc_score(y_actual_train_rfc, y_pred_train_rfc)}')


# In[31]:


y_pred_train_rfc


# In[30]:


# Create a KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and get cross-validated predictions
y_pred_train_rfc = cross_val_predict(rfc_model, x_train, y_train, cv=kf, method='predict_proba')[:, 1]

# True labels for the training set
y_actual_train_rfc = y_train

y_pred_test_rfc = rfc_model.predict_proba(x_test)[:, 1]

# True labels for the test set
y_actual_test_rfc = y_test

# Plot the ROC curve
def plot_roc_curve(true_y, y_prob, label=None):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC Curves for all features (no of features=481)')

# Plot ROC curve and print AUC score
plot_roc_curve(y_actual_train_rfc, y_pred_train_rfc, label=f'Training Set AUC: {roc_auc_score(y_actual_train_rfc, y_pred_train_rfc):.2f}')
plot_roc_curve(y_actual_test_rfc, y_pred_test_rfc, label=f'Test Set AUC: {roc_auc_score(y_actual_test_rfc, y_pred_test_rfc):.2f}')
plt.legend()
plt.savefig('ROC_Random_481_RFC.png', dpi=300)
plt.show()


# In[12]:


y_actual_train_rfc = y_train

tn1,fp1,fn1,tp1 = confusion_matrix(y_actual_train_rfc, y_pred_train_rfc).ravel()
print('RFC')
print("True Positives ::",tp1)
print("True Negatives ::",tn1)
print("False Positives ::",fp1)
print("False Negatives ::",fn1)
Accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
print("Accuracy {:0.4f}".format(Accuracy1))
Specificity1 = tn1/(tn1+fp1)
print("Specificty {:0.4f}".format(Specificity1))
Sensitivity1 = tp1/(tp1+fn1)
print("Sensitivity {:0.4f}".format(Sensitivity1))
Precision1 = tp1/(tp1+fp1)
print("Precision {:0.4f}".format(Precision1))
F1_Score1 = 2*(Precision1*Sensitivity1)/(Precision1+Sensitivity1)
print("F1_Score {:-.4f}".format(F1_Score1))
MCC1 = matthews_corrcoef(y_actual_train_rfc, y_pred_train_rfc)
print("MCC {:0.4f}".format(MCC1))
ROC1 = metrics.roc_auc_score(y_actual_train_rfc, y_pred_train_rfc)
print("ROC {:0.4f}".format(ROC1))


# In[13]:


#y_pred_test_rfc = rfc_model.predict(x_test_sc)
y_pred_test_rfc = rfc_model.predict(x_test)


# In[14]:


y_actual_test_rfc = y_test

tn1,fp1,fn1,tp1 = confusion_matrix(y_actual_test_rfc, y_pred_test_rfc).ravel()
print('RFC')
print("True Positives ::",tp1)
print("True Negatives ::",tn1)
print("False Positives ::",fp1)
print("False Negatives ::",fn1)
Accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
print("Accuracy {:0.4f}".format(Accuracy1))
Specificity1 = tn1/(tn1+fp1)
print("Specificty {:0.4f}".format(Specificity1))
Sensitivity1 = tp1/(tp1+fn1)
print("Sensitivity {:0.4f}".format(Sensitivity1))
Precision1 = tp1/(tp1+fp1)
print("Precision {:0.4f}".format(Precision1))
F1_Score1 = 2*(Precision1*Sensitivity1)/(Precision1+Sensitivity1)
print("F1_Score {:-.4f}".format(F1_Score1))
MCC1 = matthews_corrcoef(y_actual_test_rfc, y_pred_test_rfc)
print("MCC {:0.4f}".format(MCC1))
ROC1 = metrics.roc_auc_score(y_actual_test_rfc, y_pred_test_rfc)
print("ROC {:0.4f}".format(ROC1))


# In[15]:


mlp_model = MLPClassifier(random_state=42, max_iter=2000)

mlp_model.fit(x_train_sc, y_train)

# Create a KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and get cross-validated predictions
y_pred_train_mlp = cross_val_predict(mlp_model, x_train_sc, y_train, cv=kf)


# In[ ]:


cross_val_results = cross_val_score(mlp_model, x_train_sc, y_train, cv=kf)
print(f'Cross-Validation Results (Accuracy): {cross_val_results}')
print(f'Mean Accuracy: {cross_val_results.mean()}')


# In[ ]:


# Create a pipeline with scaling (optional but recommended for neural networks)
pipeline = make_pipeline(StandardScaler(), mlp_model)

# Perform cross-validation and get cross-validated predictions
y_pred_train_mlp = cross_val_predict(pipeline, x_train, y_train, cv=kf)

# Print classification report
print("Classification Report for MLP Model:")
print(classification_report(y_train, y_pred_train_mlp))

# Optionally, you can compute and print other metrics like accuracy, precision, recall, etc.
# Example:
accuracy = cross_val_score(pipeline, x_train_sc, y_train, cv=kf, scoring='accuracy')
precision = cross_val_score(pipeline, x_train_sc, y_train, cv=kf, scoring='precision')
recall = cross_val_score(pipeline, x_train_sc, y_train, cv=kf, scoring='recall')
f1 = cross_val_score(pipeline, x_train_sc, y_train, cv=kf, scoring='f1')

print(f"Accuracy: {accuracy.mean():.2f}")
print(f"Precision: {precision.mean():.2f}")
print(f"Recall: {recall.mean():.2f}")
print(f"F1 Score: {f1.mean():.2f}")


# In[16]:


y_actual_train_mlp = y_train

tn1,fp1,fn1,tp1 = confusion_matrix(y_actual_train_mlp, y_pred_train_mlp).ravel()
print('MLP')
print("True Positives ::",tp1)
print("True Negatives ::",tn1)
print("False Positives ::",fp1)
print("False Negatives ::",fn1)
Accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
print("Accuracy {:0.4f}".format(Accuracy1))
Specificity1 = tn1/(tn1+fp1)
print("Specificty {:0.4f}".format(Specificity1))
Sensitivity1 = tp1/(tp1+fn1)
print("Sensitivity {:0.4f}".format(Sensitivity1))
Precision1 = tp1/(tp1+fp1)
print("Precision {:0.4f}".format(Precision1))
F1_Score1 = 2*(Precision1*Sensitivity1)/(Precision1+Sensitivity1)
print("F1_Score {:-.4f}".format(F1_Score1))
MCC1 = matthews_corrcoef(y_actual_train_mlp, y_pred_train_mlp)
print("MCC {:0.4f}".format(MCC1))
ROC1 = metrics.roc_auc_score(y_actual_train_mlp, y_pred_train_mlp)
print("ROC {:0.4f}".format(ROC1))


# In[18]:


y_pred_test_mlp = mlp_model.predict(x_test_sc)


# In[19]:


y_actual_test_mlp = y_test

tn1,fp1,fn1,tp1 = confusion_matrix(y_actual_test_mlp, y_pred_test_mlp).ravel()
print('MLP')
print("True Positives ::",tp1)
print("True Negatives ::",tn1)
print("False Positives ::",fp1)
print("False Negatives ::",fn1)
Accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
print("Accuracy {:0.4f}".format(Accuracy1))
Specificity1 = tn1/(tn1+fp1)
print("Specificty {:0.4f}".format(Specificity1))
Sensitivity1 = tp1/(tp1+fn1)
print("Sensitivity {:0.4f}".format(Sensitivity1))
Precision1 = tp1/(tp1+fp1)
print("Precision {:0.4f}".format(Precision1))
F1_Score1 = 2*(Precision1*Sensitivity1)/(Precision1+Sensitivity1)
print("F1_Score {:-.4f}".format(F1_Score1))
MCC1 = matthews_corrcoef(y_actual_test_mlp, y_pred_test_mlp)
print("MCC {:0.4f}".format(MCC1))
ROC1 = metrics.roc_auc_score(y_actual_test_mlp, y_pred_test_mlp)
print("ROC {:0.4f}".format(ROC1))


# In[20]:


# Create an instance of XGBClassifier
Xgb_model = XGBClassifier(random_state=42)

# Fit the model on the training data
Xgb_model.fit(x_train_sc, y_train)

# Create a KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and get cross-validated predictions
y_pred_train_xgb = cross_val_predict(Xgb_model, x_train_sc, y_train, cv=kf)


# In[21]:


y_actual_train_xgb = y_train

tn1,fp1,fn1,tp1 = confusion_matrix(y_actual_train_xgb, y_pred_train_xgb).ravel()
print('XGB')
print("True Positives ::",tp1)
print("True Negatives ::",tn1)
print("False Positives ::",fp1)
print("False Negatives ::",fn1)
Accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
print("Accuracy {:0.4f}".format(Accuracy1))
Specificity1 = tn1/(tn1+fp1)
print("Specificty {:0.4f}".format(Specificity1))
Sensitivity1 = tp1/(tp1+fn1)
print("Sensitivity {:0.4f}".format(Sensitivity1))
Precision1 = tp1/(tp1+fp1)
print("Precision {:0.4f}".format(Precision1))
F1_Score1 = 2*(Precision1*Sensitivity1)/(Precision1+Sensitivity1)
print("F1_Score {:-.4f}".format(F1_Score1))
MCC1 = matthews_corrcoef(y_actual_train_xgb, y_pred_train_xgb)
print("MCC {:0.4f}".format(MCC1))
ROC1 = metrics.roc_auc_score(y_actual_train_xgb, y_pred_train_xgb)
print("ROC {:0.4f}".format(ROC1))


# In[23]:


y_pred_test_xgb = Xgb_model.predict(x_test_sc)


# In[24]:


y_actual_test_xgb = y_test

tn1,fp1,fn1,tp1 = confusion_matrix(y_actual_test_xgb, y_pred_test_xgb).ravel()
print('XGB')
print("True Positives ::",tp1)
print("True Negatives ::",tn1)
print("False Positives ::",fp1)
print("False Negatives ::",fn1)
Accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
print("Accuracy {:0.4f}".format(Accuracy1))
Specificity1 = tn1/(tn1+fp1)
print("Specificty {:0.4f}".format(Specificity1))
Sensitivity1 = tp1/(tp1+fn1)
print("Sensitivity {:0.4f}".format(Sensitivity1))
Precision1 = tp1/(tp1+fp1)
print("Precision {:0.4f}".format(Precision1))
F1_Score1 = 2*(Precision1*Sensitivity1)/(Precision1+Sensitivity1)
print("F1_Score {:-.4f}".format(F1_Score1))
MCC1 = matthews_corrcoef(y_actual_test_xgb, y_pred_test_xgb)
print("MCC {:0.4f}".format(MCC1))
ROC1 = metrics.roc_auc_score(y_actual_test_xgb, y_pred_test_xgb)
print("ROC {:0.4f}".format(ROC1))


# In[25]:


tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(y_actual_train_svm, y_pred_train_svm).ravel()

# Confusion matrix for RFC
tn_rfc, fp_rfc, fn_rfc, tp_rfc = confusion_matrix(y_actual_train_rfc, y_pred_train_rfc).ravel()

# Confusion matrix for MLP
tn_mlp, fp_mlp, fn_mlp, tp_mlp = confusion_matrix(y_actual_train_mlp, y_pred_train_mlp).ravel()

# Confusion matrix for XGB
tn_xgb, fp_xgb, fn_xgb, tp_xgb = confusion_matrix(y_actual_train_xgb, y_pred_train_xgb).ravel()

data_train = {
    'Model': ['SVM', 'RFC', 'MLP', 'XGB'],
    #'True Positives': [tp_svm, tp_rfc, tp_mlp, tp_xgb],
    #'True Negatives': [tn_svm, tn_rfc, tn_mlp, tn_xgb],
    #'False Positives': [fp_svm, fp_rfc, fp_mlp, fp_xgb],
    #'False Negatives': [fn_svm, fn_rfc, fn_mlp, fn_xgb],
    'Accuracy': [round((tp_svm+tn_svm)/(tp_svm+tn_svm+fp_svm+fn_svm), 2),
                 round((tp_rfc+tn_rfc)/(tp_rfc+tn_rfc+fp_rfc+fn_rfc), 2),
                 round((tp_mlp+tn_mlp)/(tp_mlp+tn_mlp+fp_mlp+fn_mlp), 2),
                 round((tp_xgb+tn_xgb)/(tp_xgb+tn_xgb+fp_xgb+fn_xgb), 2)],
    'Specificity': [round(tn_svm/(tn_svm+fp_svm), 2),
                    round(tn_rfc/(tn_rfc+fp_rfc), 2),
                    round(tn_mlp/(tn_mlp+fp_mlp), 2),
                    round(tn_xgb/(tn_xgb+fp_xgb), 2)],
    'Sensitivity': [round(tp_svm/(tp_svm+fn_svm), 2),
                    round(tp_rfc/(tp_rfc+fn_rfc), 2),
                    round(tp_mlp/(tp_mlp+fn_mlp), 2),
                    round(tp_xgb/(tp_xgb+fn_xgb), 2)],
    'Precision': [round(tp_svm/(tp_svm+fp_svm), 2),
                  round(tp_rfc/(tp_rfc+fp_rfc), 2),
                  round(tp_mlp/(tp_mlp+fp_mlp), 2),
                  round(tp_xgb/(tp_xgb+fp_xgb), 2)],
    'F1 Score': [round(2*(tp_svm/(tp_svm+fp_svm))*(tp_svm/(tp_svm+fn_svm))/((tp_svm/(tp_svm+fp_svm))+(tp_svm/(tp_svm+fn_svm))), 2),
                 round(2*(tp_rfc/(tp_rfc+fp_rfc))*(tp_rfc/(tp_rfc+fn_rfc))/((tp_rfc/(tp_rfc+fp_rfc))+(tp_rfc/(tp_rfc+fn_rfc))), 2),
                 round(2*(tp_mlp/(tp_mlp+fp_mlp))*(tp_mlp/(tp_mlp+fn_mlp))/((tp_mlp/(tp_mlp+fp_mlp))+(tp_mlp/(tp_mlp+fn_mlp))), 2),
                 round(2*(tp_xgb/(tp_xgb+fp_xgb))*(tp_xgb/(tp_xgb+fn_xgb))/((tp_xgb/(tp_xgb+fp_xgb))+(tp_xgb/(tp_xgb+fn_xgb))), 2)],
    'MCC': [round(matthews_corrcoef(y_actual_train_svm, y_pred_train_svm), 2),
            round(matthews_corrcoef(y_actual_train_rfc, y_pred_train_rfc), 2),
            round(matthews_corrcoef(y_actual_train_mlp, y_pred_train_mlp), 2),
            round(matthews_corrcoef(y_actual_train_xgb, y_pred_train_xgb), 2)],
    'ROC': [round(roc_auc_score(y_actual_train_svm, y_pred_train_svm), 2),
            round(roc_auc_score(y_actual_train_rfc, y_pred_train_rfc), 2),
            round(roc_auc_score(y_actual_train_mlp, y_pred_train_mlp), 2),
            round(roc_auc_score(y_actual_train_xgb, y_pred_train_xgb), 2)]
}

df_train = pd.DataFrame(data_train)
df_train


# In[26]:


tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(y_actual_test_svm, y_pred_test_svm).ravel()

# Confusion matrix for RFC
tn_rfc, fp_rfc, fn_rfc, tp_rfc = confusion_matrix(y_actual_test_rfc, y_pred_test_rfc).ravel()

# Confusion matrix for MLP
tn_mlp, fp_mlp, fn_mlp, tp_mlp = confusion_matrix(y_actual_test_mlp, y_pred_test_mlp).ravel()

# Confusion matrix for XGB
tn_xgb, fp_xgb, fn_xgb, tp_xgb = confusion_matrix(y_actual_test_xgb, y_pred_test_xgb).ravel()

data_test = {
    'Model': ['SVM', 'RFC', 'MLP', 'XGB'],
    #'True Positives': [tp_svm, tp_rfc, tp_mlp, tp_xgb],
    #'True Negatives': [tn_svm, tn_rfc, tn_mlp, tn_xgb],
    #'False Positives': [fp_svm, fp_rfc, fp_mlp, fp_xgb],
    #'False Negatives': [fn_svm, fn_rfc, fn_mlp, fn_xgb],
    'Accuracy': [round((tp_svm+tn_svm)/(tp_svm+tn_svm+fp_svm+fn_svm), 2),
                 round((tp_rfc+tn_rfc)/(tp_rfc+tn_rfc+fp_rfc+fn_rfc), 2),
                 round((tp_mlp+tn_mlp)/(tp_mlp+tn_mlp+fp_mlp+fn_mlp), 2),
                 round((tp_xgb+tn_xgb)/(tp_xgb+tn_xgb+fp_xgb+fn_xgb), 2)],
    'Specificity': [round(tn_svm/(tn_svm+fp_svm), 2),
                    round(tn_rfc/(tn_rfc+fp_rfc), 2),
                    round(tn_mlp/(tn_mlp+fp_mlp), 2),
                    round(tn_xgb/(tn_xgb+fp_xgb), 2)],
    'Sensitivity': [round(tp_svm/(tp_svm+fn_svm), 2),
                    round(tp_rfc/(tp_rfc+fn_rfc), 2),
                    round(tp_mlp/(tp_mlp+fn_mlp), 2),
                    round(tp_xgb/(tp_xgb+fn_xgb), 2)],
    'Precision': [round(tp_svm/(tp_svm+fp_svm), 2),
                  round(tp_rfc/(tp_rfc+fp_rfc), 2),
                  round(tp_mlp/(tp_mlp+fp_mlp), 2),
                  round(tp_xgb/(tp_xgb+fp_xgb), 2)],
    'F1 Score': [round(2*(tp_svm/(tp_svm+fp_svm))*(tp_svm/(tp_svm+fn_svm))/((tp_svm/(tp_svm+fp_svm))+(tp_svm/(tp_svm+fn_svm))), 2),
                 round(2*(tp_rfc/(tp_rfc+fp_rfc))*(tp_rfc/(tp_rfc+fn_rfc))/((tp_rfc/(tp_rfc+fp_rfc))+(tp_rfc/(tp_rfc+fn_rfc))), 2),
                 round(2*(tp_mlp/(tp_mlp+fp_mlp))*(tp_mlp/(tp_mlp+fn_mlp))/((tp_mlp/(tp_mlp+fp_mlp))+(tp_mlp/(tp_mlp+fn_mlp))), 2),
                 round(2*(tp_xgb/(tp_xgb+fp_xgb))*(tp_xgb/(tp_xgb+fn_xgb))/((tp_xgb/(tp_xgb+fp_xgb))+(tp_xgb/(tp_xgb+fn_xgb))), 2)],
    'MCC': [round(matthews_corrcoef(y_actual_test_svm, y_pred_test_svm), 2),
            round(matthews_corrcoef(y_actual_test_rfc, y_pred_test_rfc), 2),
            round(matthews_corrcoef(y_actual_test_mlp, y_pred_test_mlp), 2),
            round(matthews_corrcoef(y_actual_test_xgb, y_pred_test_xgb), 2)],
    'ROC': [round(roc_auc_score(y_actual_test_svm, y_pred_test_svm), 2),
            round(roc_auc_score(y_actual_test_rfc, y_pred_test_rfc), 2),
            round(roc_auc_score(y_actual_test_mlp, y_pred_test_mlp), 2),
            round(roc_auc_score(y_actual_test_xgb, y_pred_test_xgb), 2)]
}

df_test = pd.DataFrame(data_test)
df_test


# In[ ]:


df2=pd.

