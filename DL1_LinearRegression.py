#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[2]:



df = pd.read_csv('HousingData.csv')

# Display first 5 rows
print(df.head())


# In[3]:


print(df.info())
print(df.shape)


# In[4]:


print(df.isnull().sum())


# In[5]:


df = df.fillna(df.mean())


# In[6]:


X = df.drop('MEDV', axis=1)
y = df['MEDV']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# In[8]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

# Hidden Layer 2
model.add(Dense(32, activation='relu'))

# Output Layer
model.add(Dense(1))


# In[10]:


model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)


# In[11]:


model.summary()


# In[12]:


history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)


# In[13]:


loss, mae = model.evaluate(X_test, y_test)

print("Loss:", loss)
print("MAE:", mae)


# In[14]:


predictions = model.predict(X_test)

print(predictions[:5])


# In[15]:


rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("RMSE:", rmse)


# In[16]:


plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')

plt.show()


# In[ ]:




