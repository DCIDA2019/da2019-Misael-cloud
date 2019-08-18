#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Introduction to NumPy


# In[1]:


import numpy
numpy.__version__


# In[2]:


import numpy as np


# ### Reminder about Built In Documentation

# In[ ]:


np.<TAB>
#se desplegaron varias opciones


# In[3]:


get_ipython().run_line_magic('pinfo', 'np')


# ## Understanding Data Types in Python
# 

# ### A Python List Is More Than Just a List

# In[5]:


L = list(range(10))
L


# In[6]:


type(L[0])


# In[7]:


L2 = [str(c) for c in L]
L2


# In[8]:


type(L2[0])


# In[9]:


L3 = [True, "2", 3.0, 4]
[type(item) for item in L3]


# ### Fixed-Type Arrays in Python

# In[10]:


import array
L = list(range(10))
A = array.array('i', L)
A


# In[11]:


import numpy as np


# ### Creating Arrays from Python Lists

# In[12]:


# arreglo entero:
np.array([1, 4, 2, 5, 3])


# In[13]:


np.array([3.14, 4, 2, 3])


# In[14]:


np.array([1, 2, 3, 4], dtype='float32')


# In[15]:


# nested lists result in multi-dimensional arrays
np.array([range(i, i + 3) for i in [2, 4, 6]])


# ### Creating Arrays from Scratch

# In[16]:


# Crea un arreglo de enteros de longitud-10 lleno de ceros
np.zeros(10, dtype=int)


# In[17]:


# Crea un arreglo de punto-flotante de  3x5 lleno de unos
np.ones((3, 5), dtype=float)


# In[18]:


# Crea un arreglo de  3x5 lleno de puros 3.14
np.full((3, 5), 3.14)


# In[19]:


np.arange(0, 20, 2)


# In[20]:


np.linspace(0, 1, 5)


# In[21]:


# Crea  un arreglo de 3x3 con números aleatorios entre 0 y 1
np.random.random((3, 3))


# In[22]:


# Crea un arreglo de 3x3 de valores aleatorios normalmente distribuidos. 
# con media 0 and desviacion estándar 1
np.random.normal(0, 1, (3, 3))


# In[23]:


# Crea un arreglo de 3x3 de número enteros en el intervalo [0, 10)
np.random.randint(0, 10, (3, 3))


# In[24]:


# Crea una matriz identidad de  3x3
np.eye(3)


# In[25]:


# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)


# ### NumPy Standard Data Types

# In[31]:


np.zeros(10, dtype='complex64')


# In[30]:


#otra forma
np.zeros(10, dtype=np.complex64)


# ## The Basics of NumPy Arrays

# ### NumPy Array Attributes

# In[47]:


import numpy as np
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(6, size=3)  # One-dimensional array
x2 = np.random.randint(6, size=(5, 6))  # Two-dimensional array
x3 = np.random.randint(6, size=(5, 3, 8 ))  # Three-dimensional array


# In[48]:


#Each array has attributes ndim (the number of dimensions), shape (the size of each dimension), and size (the total size of the array):
print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)


# In[49]:


print("dtype:", x3.dtype)


# In[50]:


print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")


# ### Array Indexing: Accessing Single Elements

# In[51]:


x1


# In[52]:


x1[0]


# In[53]:


x1[2]


# In[54]:


x1[-1]


# In[55]:


x1[-2]


# In[56]:


x2


# In[57]:


x2[0, 0]


# In[58]:


x2[2, 0]


# In[59]:


x2[2, -1]


# In[60]:


x2[0, 0] = 5
x2


# In[61]:


x1[0] = 3.14159
x1


# ### Array Slicing: Accessing Subarrays

# In[63]:


#x[start:stop:step]


# #### One-dimensional subarrays

# In[64]:


x = np.arange(10)
x


# In[65]:


x[:5]  # first five elements


# In[66]:


x[5:]  # elements after index 5


# In[67]:


x[4:7]  # middle sub-array


# In[68]:


x[::2]   #cada elemento que esté en una posición par


# In[70]:


x[1::2] #cada elemento que esté en una posición par empezando desde la posición uno


# In[71]:


x[::-1]  # all elements, reversed


# In[72]:


x[5::-2]  # reversed every other from index 5


# #### Multi-dimensional subarrays

# In[73]:


x2


# In[74]:


x2[:2, :3]  # two rows, three columns


# In[75]:


x2[:3, ::2]  # all rows, every other column


# In[76]:


x2[::-1, ::-1]


# ##### Accessing array rows and columns

# In[77]:


print(x2[:, 0])  # first column of x2


# In[78]:


print(x2[0, :])  # first row of x2


# In[79]:


#otra forma en la sintaxis de la linea anterior 
print(x2[0])  # equivalent to x2[0, :]


# #### Subarrays as no-copy views

# In[80]:


print(x2)


# In[81]:


#Let's extract a 2×2 subarray from this

x2_sub = x2[:2, :2]
print(x2_sub)


# In[82]:


x2_sub[0, 0] = 65
print(x2_sub)


# In[83]:


#ya se ha modificado el arreglo x2
print(x2)


# #### Creating copies of arrays
# 

# In[84]:


x2_sub_copia = x2[:2, :2].copy()
print(x2_sub_copia)


# In[86]:


x2_sub_copia[0, 0] = 78
print(x2_sub_copia)


# In[87]:


print(x2)


# ### Reshaping of Arrays

# In[88]:


arr = np.arange(1, 10).reshape((3, 3))
print(arr)


# In[89]:


x = np.array([1, 2, 3])

# row vector via reshape
x.reshape((1, 3))


# In[90]:


# row vector via newaxis
x[np.newaxis, :]


# In[91]:


# column vector via reshape
x.reshape((3, 1))


# In[92]:


# column vector via newaxis
x[:, np.newaxis]


# ### Array Concatenation and Splitting

# #### Concatenation of arrays

# In[93]:


x = np.array([4, 5, 6])
y = np.array([6, 5, 4])
np.concatenate([x, y])


# In[94]:


z = [7, 7, 9]
print(np.concatenate([x, y, z]))


# In[97]:


arr = np.array([[7, 5, 5],
                 [2, 3, 9]])


# In[98]:


# concatenate along the first axis
np.concatenate([arr, arr])


# In[99]:


# concatenate along the second axis (zero-indexed)
np.concatenate([arr, arr], axis=1)


# In[100]:


x = np.array([1, 2, 3])
arr = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, arr])


# In[101]:


# horizontally stack the arrays
y = np.array([[65],
              [7]])
np.hstack([arr, y])


# #### Splitting of arrays

# In[106]:


x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)


# In[109]:


arr = np.arange(16).reshape((4, 4))
arr


# In[111]:


upper, lower = np.vsplit(arr, [2])
print(upper)
print(lower)


# In[113]:


left, right = np.hsplit(arr, [2])
print(left)
print(right)
#Similarly, np.dsplit will split arrays along the third axis


# ## Computation on NumPy Arrays: Universal Functions

# ### The Slowness of Loops

# In[117]:


import numpy as np
np.random.seed(0)

def calculo_de_reciprocos(valores):
    salida = np.empty(len(valores))
    for i in range(len(valores)):
        salida[i] = 3.5 / valores[i]
    return salida
        
valores = np.random.randint(1, 10, size=8)
calculo_de_reciprocos(valores)


# In[118]:


big_array = np.random.randint(1, 100, size=1000000)
get_ipython().run_line_magic('timeit', 'compute_reciprocals(big_array)')


# ### Introducing UFuncs

# In[119]:


print(calculo_de_reciprocos(valores))
print(3.5 / valores)


# In[120]:


get_ipython().run_line_magic('timeit', '(3.5 / big_array)')


# In[123]:


np.arange(4) / np.arange(1, 5)


# In[125]:


x = np.arange(9).reshape((3, 3))
4 ** x


# ### Exploring NumPy's UFuncs

# ##### Array arithmetic

# In[127]:


x = np.arange(5)
print("x     =", x)
print("x + 2 =", x + 2)
print("x - 2 =", x - 2)
print("x * 4 =", x * 4)
print("x / 3 =", x / 3)
print("x // 4 =", x // 4)  # floor division


# In[129]:


print("-x     = ", -x)
print("x ** 3 = ", x ** 3)
print("x % 3  = ", x % 3)


# In[130]:


-(3.5*x + 4) ** 3


# In[132]:


np.add(x, 4)


# In[133]:


#Otra operacion 
np.floor_divide(x,2)


# In[134]:


np.mod(x,2)


# In[135]:


np.multiply(x,2)


# #### Absolute value

# In[136]:


x = np.array([-2, -3, -6, 4, 2])
abs(x)


# In[137]:


np.absolute(x)


# In[138]:


np.abs(x)


# In[139]:


x = np.array([1 - 2j, 4 - 5j, 8 + 9j, 0 + 4j])
np.abs(x)


# #### Trigonometric functions

# In[140]:


#definiendo un arreglo de ángulos
theta = np.linspace(0, np.pi, 5)


# In[141]:


print("theta      = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))


# In[142]:


x = [-0.5, 0, 0.5]
print("x         = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))


# #### Exponents and logarithms

# In[144]:


x = [2, 3, 4]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("5^x   =", np.power(5, x))


# In[147]:


x = [1, 3, 5, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))


# In[148]:


x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))


# #### Specialized ufuncs

# In[149]:


from scipy import special


# In[150]:


# Gamma functions (generalized factorials) and related functions
x = [3, 7, 11]
print("gamma(x)     =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)   =", special.beta(x, 2))


# In[151]:


# Error function (integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.5, 0.9, 1.5])
print("erf(x)  =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))


# ### Advanced Ufunc Features

# #### Specifying output

# In[153]:


x = np.arange(4)
y = np.empty(4)
np.multiply(x, 11, out=y)
print(y)


# In[156]:


y = np.zeros(8)
np.power(2, x, out=y[::2])
print(y)


# #### Aggregates

# In[157]:


x = np.arange(1, 5)
np.add.reduce(x)


# In[158]:


np.multiply.reduce(x)


# In[159]:


np.add.accumulate(x)


# In[160]:


np.multiply.accumulate(x)


# #### Outer products

# In[161]:


x = np.arange(1, 8)
np.multiply.outer(x, x)


# ## Aggregations: Min, Max, and Everything In Between

# ### Summing the Values in an Array

# In[162]:


import numpy as np


# In[163]:


A = np.random.random(120)
sum(A)


# In[164]:


np.sum(A)


# In[166]:


b_arr = np.random.rand(1000000)
get_ipython().run_line_magic('timeit', 'sum(b_arr)')
get_ipython().run_line_magic('timeit', 'np.sum(b_arr)')


# ### Minimum and Maximum

# In[167]:


min(b_arr), max(b_arr)


# In[168]:


np.min(b_arr), np.max(b_arr)


# In[169]:


get_ipython().run_line_magic('timeit', 'min(b_arr)')
get_ipython().run_line_magic('timeit', 'np.min(b_arr)')


# In[170]:


print(b_arr.min(), b_arr.max(), b_arr.sum())


# #### Multi dimensional aggregates

# In[171]:


B = np.random.random((3, 4))
print(B)


# In[172]:


B.sum()


# In[173]:


B.min(axis=0)


# In[175]:


B.max(axis=1)


# #### Other aggregation functions

# In[176]:


#algunas funciones


# ## Computation on Arrays: Broadcasting

# ### Introducing Broadcasting

# In[179]:


import numpy as np


# In[180]:


a = np.array([1, 2, 3])
b = np.array([3, 4, 5])
a + b


# In[181]:


a + 8


# In[182]:


A = np.ones((3, 3))
A


# In[183]:


A + a


# In[184]:


a = np.arange(4)
b = np.arange(4)[:, np.newaxis]

print(a)
print(b)


# In[185]:


a + b


# ### Rules of Broadcasting

# #### Broadcasting example 1

# In[187]:


A = np.ones((3, 5))
a = np.arange(5)


# In[188]:


A + a


# #### Broadcasting example 2

# In[194]:


a = np.arange(3).reshape((3, 1))
b = np.arange(3)


# In[195]:


a + b


# #### Broadcasting example 3

# In[201]:


M = np.ones((3, 2))
a = np.arange(3)


# In[203]:


a[:, np.newaxis].shape


# In[205]:


M + a[:, np.newaxis]


# In[206]:


np.logaddexp(M, a[:, np.newaxis])


# ### Broadcasting in Practice

# #### Centering an array

# In[207]:


X = np.random.random((10, 3))


# In[208]:


Xmean = X.mean(0)
Xmean


# In[209]:


X_centered = X - Xmean


# In[211]:


X_centered.mean(1)


# #### Plotting a two-dimensional function

# In[215]:


# x and y have 30 steps from 0 to 5, lo dejé en 5 por el tiempo
x = np.linspace(0, 5, 30)
y = np.linspace(0, 5, 30)[:, np.newaxis]

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


# In[216]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[217]:


plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],
           cmap='viridis')
plt.colorbar();


# ## Comparisons, Masks, and Boolean Logic

# ### Comparison Operators as ufuncs

# In[223]:


x = np.array([6, 7, 8, 9, 10])


# In[225]:


x < 8  # less than


# In[226]:


x > 8  # greater than


# In[227]:


x <= 8  # less than or equal


# In[228]:


x >= 8  # greater than or equal


# In[229]:


x != 8  # not equal


# In[230]:


x == 8  # equal


# In[231]:


(3 * x) == (x ** 3)


# In[232]:


(3 * x) < (x ** 3)


# In[233]:


rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
x


# In[234]:


x < 6


# ### Working with Boolean Arrays

# In[235]:


print(x)


# #### Counting entries

# In[237]:


# how many values less than 6?
np.count_nonzero(x < 8)


# In[238]:


np.sum(x < 8)


# In[239]:


# how many values less than 8 in each row?
np.sum(x < 8, axis=1)


# In[240]:


# are there any values greater than 8?
np.any(x > 8)


# In[241]:


# are there any values less than zero?
np.any(x < 0)


# In[242]:


# are all values less than 10?
np.all(x < 10)


# In[245]:


# are all values equal to 8?
np.all(x == 8)


# In[246]:


# are all values in each row less than 8?
np.all(x < 8, axis=1)


# ### Boolean Arrays as Masks

# In[248]:


x


# In[249]:


x < 6


# In[250]:


x[x < 6]


# ### Aside: Using the Keywords and/or Versus the Operators &/|

# In[251]:


bool(22), bool(0)


# In[252]:


bool(22 and 0)


# In[253]:


bool(22 or 0)


# In[258]:


bin(22)


# In[259]:


bin(100)


# In[260]:


bin(22 & 100)


# In[262]:


bin(22 | 100)


# In[263]:


A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A | B


# In[264]:


A or B


# In[266]:


x = np.arange(12)
(x > 5) & (x < 8)


# In[267]:


(x > 5) and (x < 8)


# ## Fancy Indexing

# ### Exploring Fancy Indexing

# In[268]:


import numpy as np
rand = np.random.RandomState(42)

x = rand.randint(100, size=12)
print(x)


# In[269]:


[x[2], x[8], x[4]]


# In[271]:


ind = [2, 8, 4]
x[ind]


# In[272]:


ind = np.array([[1, 2],
                [3, 4]])
x[ind]


# In[274]:


X = np.arange(12).reshape((3, 4))
X


# In[276]:


row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]


# In[277]:


X[row[:, np.newaxis], col]


# In[278]:


row[:, np.newaxis] * col


# ### Combined Indexing

# In[279]:


print(X)


# In[280]:


X[2, [2, 0, 1]]


# In[281]:


X[1:, [2, 0, 1]]


# ## Structured Data: NumPy's Structured Arrays

# In[9]:


import numpy as np


# In[11]:


name = ['Manuel', 'Bárbara', 'Ramona', 'Luisa']
age = [22, 35, 67, 29]
weight = [55.0, 85.5, 68.0, 61.5]


# In[13]:


x = np.zeros(4, dtype=int)


# In[14]:


# Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)


# In[15]:


data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)


# In[16]:


# Get all names
data['name']


# In[17]:


# Get first row of data
data[0]


# In[18]:


# Get the name from the last row
data[-1]['name']


# In[19]:


# Get names where age is under 30
data[data['age'] < 40]['name']


# ### Creating Structured Arrays

# In[20]:


np.dtype({'names':('name', 'age', 'weight'),
          'formats':('U10', 'i4', 'f8')})


# In[21]:


np.dtype({'names':('name', 'age', 'weight'),
          'formats':((np.str_, 10), int, np.float32)})


# In[22]:


np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])


# In[23]:


np.dtype('S10,i4,f8')


# In[24]:


np.dtype('u1') == np.uint8


# In[25]:


np.dtype('c16') == np.complex128


# In[26]:


np.dtype('S5')


# In[27]:


np.dtype('V') == np.void


# ### More Advanced Compound Types

# In[28]:


tp = np.dtype([('id', 'i8'), ('mat', 'f8', (4, 4))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])


# ### RecordArrays: Structured Arrays with a Twist

# In[29]:


data['age']


# In[30]:


data_rec = data.view(np.recarray)
data_rec.age


# In[31]:


get_ipython().run_line_magic('timeit', "data['age']")
get_ipython().run_line_magic('timeit', "data_rec['age']")
get_ipython().run_line_magic('timeit', 'data_rec.age')


# In[ ]:




