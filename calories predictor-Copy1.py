#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


calories = pd.read_csv("C:/Users/HP/Desktop/python prodect/calories 1.csv")
print(calories)


# In[3]:


exercise = pd.read_csv("C:/Users/HP/Desktop/python prodect/exercise.csv")
print(exercise)


# In[4]:


calories.head(2)


# In[5]:


exercise.head(2)


# In[6]:


df = exercise.merge(calories,on='User_ID')


# In[7]:


df.head(3)


# In[8]:


plt.hist(df['Gender'],bins=12)


# In[9]:


df.shape


# In[10]:


df['Gender'].value_counts()


# In[11]:


plt.hist(df['Age'],bins=20)


# In[12]:


sns.scatterplot(x=df['Calories'],y=df['Duration'])


# In[13]:


sns.barplot(x=df["Calories"],y=df['Gender'])


# In[14]:


sns.barplot(x=df['Gender'],y=df['Age'])


# In[15]:


sns.boxplot(x=df['Gender'],y=df['Age'])


# In[16]:


df.describe()


# In[17]:


sns.distplot(df['Age'])


# In[18]:


sns.barplot(x=df['Age'],y=df['Gender'])


# In[19]:


sns.lineplot(x=df['Age'],y=df['Gender'])


# In[ ]:





# In[20]:


df.head(3)


# In[21]:


#train test split


# In[22]:


X=df.drop(['User_ID','Calories'],axis=1)
Y=df['Calories']


# In[23]:


X.shape


# In[24]:


Y.shape


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[27]:


X_train.shape


# In[28]:


X_test.shape


# In[29]:


#Training model


# In[30]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error


# In[31]:


models = {'lr':LinearRegression(),
          'rd':Ridge(),
          'ls':Lasso(),
          'dtr':DecisionTreeRegressor(),
          'rfr':RandomForestRegressor()
}


# In[32]:


for name, mod in models.items():
    mod.fit(X_train,Y_train)
    Y_pred = mod.predict(X_test)
    print(f"{name} MSE: {mean_squared_error(Y_test,Y_pred)},score: {r2_score(Y_test,Y_pred)}")


# In[ ]:


rfr = RandomForestRegressor()
rfr.fit(X_train,Y_train)
Y_pred = rfr.predict(X_test)


# In[ ]:


import pickle
pickle.dump(rfr,open('rfr.pkl','wb'))


# In[ ]:


X_train.to_csv('X_train.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




