#!/usr/bin/env python
# coding: utf-8

# Programa para calcular los eigenvalores y los eigenvectores de una arreglo (matriz) cuadrado.

# In[1]:


import numpy as np


# In[5]:


M = np.mat("89 -5;11 0")


# In[3]:


print("Matriz Original:")


# In[7]:


print("\n", M)


# In[8]:


w, v = np.linalg.eig(M) 


# In[9]:


print( "Eigenvalores de la matriz dada",w)


# In[10]:


print( "Eigenvalores de la matriz dada",v)


# In[ ]:




