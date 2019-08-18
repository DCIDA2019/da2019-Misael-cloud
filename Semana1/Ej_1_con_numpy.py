#!/usr/bin/env python
# coding: utf-8

# Calcular el producto cruz entre dos vectores.

# In[6]:


import numpy as np 


# In[7]:


a = [[7, 8], [9, 10]]


# In[8]:


b = [[3, 4], [5, 6]]


# In[9]:


print("Matriz Original:")


# In[10]:


print(a)


# In[12]:


print(b)


# In[13]:


res_1 = np.cross(a, b)


# In[21]:


res_2 = np.cross(b, a)


# In[15]:


print("El producto cruz entre dos vectores a y b:")


# In[16]:


print(res_1)


# In[17]:


print("El producto cruz entre dos vectores b y a:")


# In[22]:


print(res_2)


# In[ ]:




