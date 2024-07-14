from dynesty import NestedSampler
import numpy as np
import time

ndim = [20]

def prior(x):
    return -1 + x * 3

def loglike(x, dim):
    # Calculate the Rosenbrock function value in ten dimensions
    a = 1
    b = 100
    rosenbrock = sum((a - x[i])**2 + b * (x[i+1] - x[i]**2)**2 for i in range(dim - 1))
    
    # Return the negative of the Rosenbrock function value
    return -rosenbrock

# initialize our nested sampler
with open('runtime20.csv', 'w') as file:
    file.write("Dimension,Elapsed_Time,log(Z)\n")  # Write the header



for i, dim in enumerate(ndim):
    sampler = NestedSampler(lambda x: loglike(x, dim), prior, dim)
    start_time = time.time()
    sampler.run_nested()
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    results = sampler.results
    print(results.summary())
    print(results.logz[-1], results.logzerr[-1])
    print(elapsed_time, " seconds")
    # Write the dimension and elapsed time to the file
    with open('runtime.csv', 'a') as file:
        file.write(f"{dim},{elapsed_time},{results.logz[-1]},+/-,{results.logzerr[-1]}\n")

