
# coding: utf-8

# # Import Keras as set cntk as backend

# In[17]:

import os
os.environ['KERAS_BACKEND'] = 'cntk'
import keras
import cntk


# # read the csv file

# In[18]:

import pandas as pd
data = pd.read_csv('sample-data.csv')


# ## view head of csv

# In[19]:

data.head()


# ## view tail of csv

# In[20]:

data.tail()


# ### View sample of csv

# In[21]:

data.sample()


# # Setup the model

# describe the data

# In[24]:

data.describe()


# In[25]:



from keras.models import Sequential
model = Sequential()
from keras.layers import Dense
model.add(Dense(12, activation='relu', input_shape=(11,)))


# Use SkLearn to split the data

# In[29]:

import sklearn
#from sklearn.model_selection import train_test_split 
from sklearn.cross_validation import train_test_split


# In[38]:

target_column = ['Defaulter'] 
predictors = list(set(list(data.columns))-set(target_column))
predictors


# In[34]:

df_train, df_test = train_test_split(data, train_size=0.8, test_size=0.2)
print(df_train.shape); print(df_test.shape)


# In[35]:

df_train


# In[36]:

df_test


# In[ ]:



