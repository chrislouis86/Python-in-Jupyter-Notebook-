#!/usr/bin/env python
# coding: utf-8

# ## Creating plots in python
# 
# In this notebook we look at at how to plot arrays of numbers. 
# 
# Arrays are particulalry usefull in audio, as this is how audio is represented in a computer. 
# 
# There are specific python modules and packages that we can use for this purpose. 
# 

# ##### What are modules and packages?
# 
# Python modules and packages contain code we can re-use in our notebooks, such as function, classes and variable definitions. 
# 
# Packages are not exactly the same as modules, but they share many similarities and they both need to be imported in a notebook before they can be used.
# 
# An alias is often assinged to the imported module or package that replaces temporarily their actual name in the notebook. However, caution is required, as aliases can make the code less readable!
# 
# 
# In this notebook, we will have a look at NumPy and the plot module from matplotlib, probably the most common imports among scientists. 

# In[1]:


# here is how we import packages and modules and give them aliases

import numpy as np #the alias np is assinged to numpy             
import matplotlib.pyplot as plt #the alias plt is assigned to pyplot

# each package or module can only be imported once in a notebook. 
# re-running the import has no effect!


# In[2]:


# we navigate through the imported packages similalry to how we navigate through our file system and access files
# but using '.' instead of '/'
# here is the 'path' to the function randint() of the numpy and the function plot() of pyplot

y = np.random.randint(0,100,1000)  # it generates an array of random numbers
plt.plot(y) # it plots an array


# In[3]:


# there are numerous plot properties that we can specify

# here are just a couple of examples:

plt.figure(figsize=(10,5)) # the size of the figure
plt.plot(y)                
plt.xlabel('time')         # the horizontal and vertical axis labels
plt.ylabel('values')


# In[4]:


# multiple arrays can be ploted in a single plot

y1 = 0.5*np.random.random(1000)

# the function linspace creates an array of evenly spaced numbers between the start and stop values
y2 = np.linspace(start = 0, stop = 3,num = 1000) 

y3 = y1 + y2
plt.plot(y1)
plt.plot(y2)
plt.plot(y3)


# In[5]:


# passing two arrays to plot() treats them as (x,y) values

x = np.linspace(0, 3.2, 1000)
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)

plt.show() # in python a plot is not shown automatically. the show() is needed.

# but in a jupyter notebook the figure is shown automatically
# notice however, that no [xx] is produced when explicitly using show()

