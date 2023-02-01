#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:/Users/HP/Downloads/Social_Network_Ads (1).csv")


# In[3]:


df.head()


# # Split Feature and Target

# In[4]:


X = df.iloc[:,[2,3]].values
y = df["Purchased"]


# In[5]:


X


# In[6]:


df.head(4)


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.scatter(X[y==0,0],X[y==0,1],label="Not purchased",color="red")
plt.scatter(X[y==1,0],X[y==1,1],label="Purchased",color="green")
plt.xlabel("Age")
plt.ylabel("Esitmated Salary")
plt.legend()
plt.show()


# # Standarization

# In[9]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(X)


# In[10]:


x


# # Train & Test Split

# In[11]:


df.shape


# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.75,random_state=0)


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


y_train.shape


# # Model Creation

# In[16]:


from sklearn.tree  import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)


# # Model Validation

# In[17]:


y_pred = model.predict(X_test)
y_pred1 = model.predict(X_train)


# # Model Validation Confusion Matrix

# In[18]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[19]:


accuracy_score(y_train,y_pred1)


# In[20]:


accuracy_score(y_test,y_pred)


# # Model Visulization

# ### Test Data Visulization|

# In[21]:


from sklearn import tree
plt.figure(figsize=(25,15))
tree.plot_tree(model,filled=True)

plt.show()


# In[22]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test


X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))


plt.scatter(X_set[y_set==0,0],X_set[y_set==0,1],label="Not purchased",color="red")
plt.scatter(X_set[y_set==1,0],X_set[y_set==1,1],label="Purchased",color="green")
plt.xlabel("Age")
plt.ylabel("Esitmated Salary")
plt.legend()
plt.show()


# ### Train Data Visualization

# In[23]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train


X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))


plt.scatter(X_set[y_set==0,0],X_set[y_set==0,1],label="Not purchased",color="red")
plt.scatter(X_set[y_set==1,0],X_set[y_set==1,1],label="Purchased",color="green")
plt.xlabel("Age")
plt.ylabel("Esitmated Salary")
plt.legend()
plt.show()


# # Visulization of trainging dataset

# ## Pruning Concept

# In[24]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = "gini",ccp_alpha=0.0)
model.fit(X_train,y_train)


# In[25]:


ccp = model.cost_complexity_pruning_path(X_train,y_train)
ccp


# In[32]:


ccp_alphas = ccp.ccp_alphas


# In[33]:


ccp_alphas


# In[37]:


train_score = []
test_score = []

for i in ccp_alphas:
    model1 = DecisionTreeClassifier(ccp_alpha=i)
    model1.fit(X_train,y_train)
    train_score.append(model1.score(X_train,y_train))
    test_score.append(model1.score(X_test,y_test))


# In[39]:


plt.scatter(ccp_alphas,train_score,color="green",label="Training")
plt.plot(ccp_alphas,train_score,color="green",label="Training")
plt.scatter(ccp_alphas,test_score,color="red",label="Test")
plt.plot(ccp_alphas,test_score,color="red",label="Test")
plt.xlabel("CCP alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[42]:


plt.scatter(ccp_alphas,train_score,color="green",label="Training")
plt.plot(ccp_alphas,train_score,color="green",label="Train")
plt.scatter(ccp_aplhas,test_score,color="red",label="Test")
plt.plot(ccp_alphas,test_score,color="red",label="Test")
plt.xlabel("CCP alpha")
plt.ylabel("Accuracy")
plt.xlim(0,0.025)
plt.legend()
plt.grid()
plt.show()


# In[47]:


plt.scatter(ccp_alphas,train_score,color="green",label="Training")
plt.plot(ccp_alphas,train_score,color="green",label="Training")
plt.scatter(ccp_alphas,test_score,color = "red",label="Test")
plt.plot(ccp_alphas,test_score,color = "red",label="Test")
plt.xlabel("CCP alpha")
plt.ylabel("Accuracy")
plt.xlim(0,0.010)
plt.legend()
plt.grid()
plt.show()


# In[48]:


ccp_alphas


# # Final Model

# In[49]:


final_model = DecisionTreeClassifier(ccp_alpha=0.00384921,max_depth=None,criterion="gini")
final_model.fit(X_train,y_train)


# In[50]:


y_pred=final_model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[53]:


y_pred1 = final_model.predict(X_train)
accuracy_score(y_train,y_pred1)


# In[54]:


from sklearn import tree
plt.figure(figsize=(25,15))
tree.plot_tree(final_model,filled=True)
plt.show()


# In[ ]:




