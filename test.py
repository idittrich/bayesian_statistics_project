from dynesty import NestedSampler
import numpy as np
import time

ndim = 3

def prior(x):
    return -1 + x * 3

def loglike(x):
    # Calculate the Rosenbrock function value in ten dimensions
    a = 1
    b = 100
    rosenbrock = sum((a - x[i])**2 + b * (x[i+1] - x[i]**2)**2 for i in range(ndim - 1))
    
    # Return the negative of the Rosenbrock function value
    return -rosenbrock

# initialize our nested sampler
sampler = NestedSampler(loglike, prior, ndim)

start_time = time.time()
sampler.run_nested()
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

results = sampler.results

print(results.summary())
print(results.logz[-1], results.logzerr[-1])
print(elapsed_time, " seconds")

