
# coding: utf-8

# In[32]:


# https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb
import GPy
#GPy.plotting.change_plotting_library('plotly')
GPy.plotting.change_plotting_library('matplotlib')


# 
# - Use GP
# - implement GP-UCB
# - Run on simple benchmark function
# - visualize
# - Built NN with bayesian linear regressor (run on benchmark function)
# 
# - Writeup
#     - BO approximations
#     - 

# In[46]:


import numpy as np


# In[54]:


X = np.random.uniform(-3.,3.,(1,1))
Y = np.sin(X) + np.random.randn(1,1)*0.05


# In[60]:


X.shape


# In[56]:


kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)


# In[57]:


m = GPy.models.GPRegression(X,Y,kernel)


# In[58]:


from IPython.display import display
display(m)


# In[59]:


fig = m.plot()
GPy.plotting.show(fig, filename='basic_gp_regression_notebook')


# In[40]:


m.optimize(messages=True)


# In[42]:


m.optimize_restarts(num_restarts = 10)


# In[44]:


display(m)
fig = m.plot()
GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized')


# In[45]:


# sample inputs and outputs
X = np.random.uniform(-3.,3.,(50,2))
Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05

# define kernel
ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)

# create simple GP model
m = GPy.models.GPRegression(X,Y,ker)

# optimize and plot
m.optimize(messages=True,max_f_eval = 1000)
fig = m.plot()
display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))
display(m)

