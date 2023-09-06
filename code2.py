#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import pandas as pd
full_data = pd.read_csv('data.csv')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as met
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


full_data


# In[5]:


full_data.head()


# In[6]:


full_data.shape


# In[7]:


full_data.info


# In[8]:


full_data.nunique()


# In[9]:


full_data.isnull().sum()


# In[10]:


full_data.corr()


# In[11]:


plt.figure(figsize=(10,10))
sns.heatmap(full_data.corr())


# In[12]:


full_data['gender'].unique()


# In[13]:


gender = {'M': 1,'F': 2}
full_data.gender = [gender[item] for item in full_data.gender]


# In[14]:


full_data['gender'].unique()


# In[15]:


full_data['PlaceofBirth'].unique()


# In[16]:


full_data['NationalITy'].unique()


# In[17]:


full_data.drop('NationalITy',axis=1,inplace=True)


# In[18]:


PlaceofBirth = {'KuwaIT': 1,'lebanon': 2,'Egypt':3,'SaudiArabia':4,'USA':5,'Quran':6,'Jordan':7,'venzuela':8,'Iran':9,'Tunis':10,'Morocco':11,'Syria':12,'Palestine':13,'Iraq':14,'Lybia':15}
full_data.PlaceofBirth = [PlaceofBirth[item] for item in full_data.PlaceofBirth]


# In[19]:


full_data['PlaceofBirth'].unique()


# In[20]:


full_data.shape


# In[21]:


full_data['StageID'].unique()


# In[22]:


StageID = {'lowerlevel': 1,'MiddleSchool': 2,'HighSchool':3}
full_data.StageID = [StageID[item] for item in full_data.StageID]


# In[23]:


full_data['StageID'].unique()


# In[24]:


full_data['GradeID'].unique()


# In[25]:


GradeID = {'G-04': 4,'G-07': 7,'G-08':8,'G-06':6,'G-05':5,'G-09':9,'G-12':12,'G-11':11,'G-10':10,'G-02':2,}
full_data.GradeID = [GradeID[item] for item in full_data.GradeID]


# In[26]:


full_data['GradeID'].unique()


# In[27]:


full_data['SectionID'].unique()


# In[28]:


SectionID = {'A': 1,'B': 2,'C':3}
full_data.SectionID = [SectionID[item] for item in full_data.SectionID]


# In[29]:


full_data['SectionID'].unique()


# In[30]:


full_data['Topic'].unique()


# In[31]:


Topic = {'IT': 1,'Math': 2,'Arabic':3,'Science':4,'English':5,'Quran':6,'Spanish':7,'French':8,'History':9,'Biology':10,'Chemistry':11,'Geology':12}
full_data.Topic = [Topic[item] for item in full_data.Topic]


# In[32]:


full_data['Topic'].unique()


# In[33]:


full_data['Semester'].unique()


# In[34]:


full_data['Semester'].value_counts()


# In[35]:


Semester = {'F': 1,'S': 2}
full_data.Semester = [Semester[item] for item in full_data.Semester]


# In[36]:


full_data['Semester'].unique()


# In[37]:


full_data['Relation'].value_counts()


# In[38]:


Relation = {'Father': 0,'Mum': 1}
full_data.Relation = [Relation[item] for item in full_data.Relation]


# In[40]:


full_data['Relation'].unique()


# In[41]:


full_data['ParentAnsweringSurvey'].unique()


# In[42]:


ParentAnsweringSurvey = {'Yes': 0,'No': 1}
full_data.ParentAnsweringSurvey = [ParentAnsweringSurvey[item] for item in full_data.ParentAnsweringSurvey]


# In[43]:


full_data['ParentAnsweringSurvey'].unique()


# In[44]:


full_data['ParentschoolSatisfaction'].unique()


# In[45]:


ParentschoolSatisfaction = {'Good': 0,'Bad': 1}
full_data.ParentschoolSatisfaction = [ParentschoolSatisfaction[item] for item in full_data.ParentschoolSatisfaction]


# In[46]:


full_data['ParentschoolSatisfaction'].unique()


# In[47]:


full_data['StudentAbsenceDays'].unique()


# In[48]:


StudentAbsenceDays = {'Under-7': 0,'Above-7': 1}
full_data.StudentAbsenceDays = [StudentAbsenceDays[item] for item in full_data.StudentAbsenceDays]


# In[49]:


full_data['StudentAbsenceDays'].unique()


# In[51]:


full_data.corr()


# In[52]:


plt.figure(figsize=(10,10))
sns.heatmap(full_data.corr())


# Analysis of Data

# Logistic Regression

# 50:50 train:test ratio

# In[53]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.50,random_state=0)


# In[54]:


X_train.shape,X_test.shape


# In[55]:


y_train.shape,y_test.shape


# In[56]:


lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[57]:


pred = lr.predict(X_train)
print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_train, pred)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")


# 80:20 ratio

# In[58]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[59]:


X_train.shape,X_test.shape


# In[60]:


lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[61]:


pred = lr.predict(X_train)
print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_train, pred)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")


# 75:25 ratio

# In[62]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[63]:


X_train.shape,X_test.shape


# In[64]:


lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[65]:


pred = lr.predict(X_train)
print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_train, pred)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")


# 70:30 ratio

# In[66]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


# In[67]:


X_train.shape,X_test.shape


# In[68]:


lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[69]:


pred = lr.predict(X_train)
print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_train, pred)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")


# Decision Tree

# 50:50

# In[70]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]


# In[71]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.50,random_state=0)


# In[74]:


decisiontree=DecisionTreeClassifier(criterion="entropy").fit(X_train,y_train)


# In[75]:


decisiontreepredict=decisiontree.predict(X_test)
decisiontreepredict


# In[76]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, decisiontreepredict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, decisiontreepredict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, decisiontreepredict)}\n")


# 80:20

# In[81]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[82]:


decisiontree=DecisionTreeClassifier(criterion="entropy").fit(X_train,y_train)


# In[83]:


decisiontreepredict=decisiontree.predict(X_test)
decisiontreepredict


# In[84]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, decisiontreepredict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, decisiontreepredict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, decisiontreepredict)}\n")


# 75:25

# In[85]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[86]:


decisiontree=DecisionTreeClassifier(criterion="entropy").fit(X_train,y_train)


# In[87]:


decisiontreepredict=decisiontree.predict(X_test)
decisiontreepredict


# In[88]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, decisiontreepredict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, decisiontreepredict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, decisiontreepredict)}\n")


# 70:30

# In[89]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


# In[90]:


decisiontree=DecisionTreeClassifier(criterion="entropy").fit(X_train,y_train)


# In[91]:


decisiontreepredict=decisiontree.predict(X_test)
decisiontreepredict


# In[92]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, decisiontreepredict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, decisiontreepredict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, decisiontreepredict)}\n")


# NAIVE BAYES

# 50:50

# In[93]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.50,random_state=0)
X_train.shape,X_test.shape
gaussNb=GaussianNB()
gaussNb.fit(X_train,y_train)


# In[94]:


y_predict=gaussNb.predict(X_test)


# In[95]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, y_predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_predict)}\n")


# 80:20

# In[96]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
X_train.shape,X_test.shape
gaussNb=GaussianNB()
gaussNb.fit(X_train,y_train)


# In[97]:


y_predict=gaussNb.predict(X_test)


# In[98]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, y_predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_predict)}\n")


# 75:25

# In[99]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
X_train.shape,X_test.shape
gaussNb=GaussianNB()
gaussNb.fit(X_train,y_train)


# In[100]:


y_predict=gaussNb.predict(X_test)
print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, y_predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_predict)}\n")


# 70:30

# In[101]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
X_train.shape,X_test.shape
gaussNb=GaussianNB()
gaussNb.fit(X_train,y_train)


# In[102]:


y_predict=gaussNb.predict(X_test)
print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, y_predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_predict)}\n")


# MULTILAYER PERCEPTRON

# 50:50

# In[103]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.50,random_state=0)
multilayerperceptron=MLPClassifier(max_iter=500, activation='relu')


# In[104]:


multilayerperceptron.fit(X_train,y_train)
multilayerperceptron_Predict=multilayerperceptron.predict(X_test)


# In[105]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, multilayerperceptron_Predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, multilayerperceptron_Predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, multilayerperceptron_Predict)}\n")


# 80:20

# In[106]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
multilayerperceptron=MLPClassifier(max_iter=500, activation='relu')


# In[107]:


multilayerperceptron.fit(X_train,y_train)
multilayerperceptron_Predict=multilayerperceptron.predict(X_test)


# In[108]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, multilayerperceptron_Predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, multilayerperceptron_Predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, multilayerperceptron_Predict)}\n")


# 75:25

# In[109]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
multilayerperceptron=MLPClassifier(max_iter=500, activation='relu')


# In[110]:


multilayerperceptron.fit(X_train,y_train)
multilayerperceptron_Predict=multilayerperceptron.predict(X_test)


# In[111]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, multilayerperceptron_Predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, multilayerperceptron_Predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, multilayerperceptron_Predict)}\n")


# 70:30

# In[112]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
multilayerperceptron=MLPClassifier(max_iter=500, activation='relu')


# In[113]:


multilayerperceptron.fit(X_train,y_train)
multilayerperceptron_Predict=multilayerperceptron.predict(X_test)


# In[114]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, multilayerperceptron_Predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, multilayerperceptron_Predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, multilayerperceptron_Predict)}\n")


# KNN

# 50:50

# In[115]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.50,random_state=0)


# In[116]:


KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
KNN_predict = KNN.predict(X_test)


# In[117]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, KNN_predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, KNN_predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, KNN_predict)}\n")


# 80:20

# In[118]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[119]:


KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
KNN_predict = KNN.predict(X_test)


# In[120]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, KNN_predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, KNN_predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, KNN_predict)}\n")


# 75:25

# In[121]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[122]:


KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
KNN_predict = KNN.predict(X_test)


# In[123]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, KNN_predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, KNN_predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, KNN_predict)}\n")


# 70:30

# In[124]:


X=full_data.drop('Class',axis=1)
y=full_data["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


# In[125]:


KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
KNN_predict = KNN.predict(X_test)


# In[126]:


print("Train Result:\n================================================")
print(f"Accuracy Score: {accuracy_score(y_test, KNN_predict) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, KNN_predict)}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, KNN_predict)}\n")

