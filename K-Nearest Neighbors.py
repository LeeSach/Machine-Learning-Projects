#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#K-Nearest Neighbors is a supervised learning algorithm. Where the data is 'trained' with data points corresponding to their 
#classification. To predict the class of a given data point, it takes into account the classes of the 'K' nearest data points 
#and chooses the class in which the majority of the 'K' nearest data points belong to as the predicted class.


# In[1]:


get_ipython().system('pip install scikit-learn==0.23.1')


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('teleCust1000t.csv')
df.head()


# In[5]:


#Data Visualization and Analysis
df['custcat'].value_counts()


# In[8]:


df.hist(column='income',bins=50)


# In[9]:


df.columns


# In[10]:


X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


# In[11]:


y = df['custcat'].values
y[0:5]


# In[12]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier


# In[15]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[16]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[17]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[18]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[19]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[20]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:




