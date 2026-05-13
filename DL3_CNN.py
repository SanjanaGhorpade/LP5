#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


# In[2]:


train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')


# In[3]:


train_df.head()


# In[4]:


print("Training Data Shape:", train_df.shape)
print("Testing Data Shape:", test_df.shape)


# In[5]:


X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values


# In[6]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[7]:


X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# In[8]:


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[11]:


plt.imshow(X_train[0].reshape(28,28),cmap = 'gray')
plt.title("Sample image")
plt.show()


# In[13]:


model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))


# In[14]:


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[15]:


history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


# In[16]:


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[17]:


loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)


# In[18]:


predictions = model.predict(X_test)

print("Predicted Class:", np.argmax(predictions[0]))
print("Actual Class:", np.argmax(y_test[0]))


# In[19]:


class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

plt.imshow(X_test[0].reshape(28,28), cmap='gray')

predicted_label = class_names[np.argmax(predictions[0])]
actual_label = class_names[np.argmax(y_test[0])]

plt.title(f"Predicted: {predicted_label}\nActual: {actual_label}")

plt.show()


# In[ ]:




