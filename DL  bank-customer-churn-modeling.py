#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Prediction Using Artificial Neural Network (ANN)

# Customer churn prediction is to measure why customers are leaving a business. In this tutorial we will be looking at customer churn in Bank Turnover. We will build a deep learning model to predict the churn and use precision,recall, f1-score to measure performance of our model

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data

# In[2]:


df = pd.read_csv("Churn_Modelling.csv")
df.sample(5)


# ## First of all, drop customerID , Surname,Row Number, column as it is of no use,

# In[4]:


df.drop('CustomerId',axis='columns',inplace=True)


# In[5]:


df.drop('RowNumber',axis='columns',inplace=True)


# In[6]:


df.drop('Surname',axis='columns',inplace=True)


# In[7]:


df.dtypes


# In[8]:


df


# In[9]:


df[df.Exited==0]


# ## Data Visualization

# In[12]:


tenure_Exited_no = df[df.Exited==0].Tenure
tenure_Exited_yes = df[df.Exited==1].Tenure
plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")
blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100]
plt.hist([tenure_Exited_yes, tenure_Exited_no], rwidth=0.95, color=['green','red'],label=['Exited=Yes','Exited=No'])
plt.legend()


# In[13]:


def print_unique_col_values(df):
       for column in df:
            if df[column].dtypes=='object':
                print(f'{column}: {df[column].unique()}') 


# In[14]:


print_unique_col_values(df)


# In[16]:


df['Gender'].replace({'Female':1,'Male':0},inplace=True)


# In[19]:


df.Gender.unique()


# ## One hot encoding for categorical columns

# In[20]:


df2 = pd.get_dummies(data=df, columns=['Geography'])
df2.columns


# In[21]:


df2.sample(5)


# In[22]:


df2.dtypes


# ## transforming everything to the same scaling

# In[25]:


cols_to_scale = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])


# In[26]:


for col in df2:
    print(f'{col}: {df2[col].unique()}')


# ## Train test split

# In[27]:


X = df2.drop('Exited',axis='columns')
y = df2['Exited']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[28]:


X_train.shape


# In[29]:


X_test.shape


# In[30]:


X_train[:10]


# In[31]:


len(X_train.columns)


# ## Build a model (ANN) in tensorflow/keras¶

# In[34]:


import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(12, input_shape=(12,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200)


# In[35]:


model.evaluate(X_test, y_test)


# In[36]:


yp = model.predict(X_test)
yp[:5]


# In[37]:


y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[38]:


y_pred[:10]


# In[39]:


y_test[:10]


# In[40]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[41]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




