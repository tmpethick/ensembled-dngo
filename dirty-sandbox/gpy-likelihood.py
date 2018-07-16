#%%

import numpy as np
import matplotlib.pyplot as plt
import GPy
GPy.plotting.change_plotting_library('matplotlib')# 

from scipy.optimize import minimize

n = 20
X = np.random.uniform(-3.,3.,(n,1))
Y = np.sin(X) + np.random.randn(n,1)*0.05

def mini(f, bounds, n_starts=10):
    min_y = 1 # TODO:
    min_x = None
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_starts, bounds.shape[0])):
        res = minimize(f, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if not res.success:
            print(res)
            print(min_y)
            print(min_x)
            print(x0)
            raise Exception(str(res))
        if res.fun < min_y:
            min_y = res.fun
            min_x = res.x
            
    return min_x

def log_marg_likelihood(params):
    kernel = GPy.kern.RBF(input_dim=1, variance=params[0], lengthscale=params[1])
    m = GPy.models.GPRegression(X,Y,kernel, noise_var=params[2])
    # n = X.shape[0]
    # K = kernel.K(X,X)
    # return 0.5 * np.linalg.slogdet(K)[1] \
    #     - 0.5 * np.dot(np.dot(Y.transpose(), np.linalg.inv(K)), Y)[0,0]\
    #     - (n/2) * np.log(2*np.pi)
    return m.log_likelihood()

#%%
f = lambda x: -log_marg_likelihood(x)
bounds = np.array([[0.0001,10], [0.0001,10], [0.0001,10]])
res = minimize(f, x0=np.array([0.2, 1, 0.0001]), bounds=bounds, method='L-BFGS-B')        
params = res.x
params
#%%
kernel = GPy.kern.RBF(input_dim=1, variance=params[0], lengthscale=params[1])
m = GPy.models.GPRegression(X,Y,kernel, noise_var=params[2])
m.plot()

#%%
params = mini(f, bounds=bounds)
kernel = GPy.kern.RBF(input_dim=1, variance=params[0], lengthscale=params[1])
m = GPy.models.GPRegression(X,Y,kernel, noise_var=params[2])
m.plot()
print(params)

#%%
kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1.)
m = GPy.models.GPRegression(X,Y,kernel)
fig = m.plot()

#%%
-

#%%

# xs = np.arange(0.1,10)
# plt.plot(xs, [f(x) for x in xs])


# - simple 1d example 
#   - maybe with bump (see gpy examples)
#   - Implement simple BO
#       - marginalize acq function over hyperparameters
#       - Slice sampling
# - Run BO and see it fail
# - Try spearmint
# - Try BNN
# - Implement NN
#   - Create network
#   - Replace last layer with
#   - Find training data
# 

# Questions
# - Main problem happens in high dimensions (curse of dimensionality). Importance of good prior. Should I investigate this?
# - Parallizability

