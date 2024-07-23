from dynesty import NestedSampler, DynamicNestedSampler
import time

ndim = [3]


def prior(x):
    return -1 + x * 3


def loglike(x, dim):
    a = 1
    b = 100
    rosenbrock = sum((a - x[i])**2 + b * (x[i+1] - x[i]**2)**2 for i in range(dim - 1))
    return -rosenbrock


for i, dim in enumerate(ndim):
    sampler = NestedSampler(lambda x: loglike(x, dim), prior, dim, nlive=1000)
    start_time = time.time()
    sampler.run_nested()
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    results = sampler.results

    dsampler = DynamicNestedSampler(lambda x: loglike(x, dim), 
                                    prior, 
                                    dim,  
                                    bound='single')
    dsampler.run_nested(dlogz_init=0.05, nlive_init=500, nlive_batch=100,
                        maxiter_init=10000, maxiter_batch=1000, maxbatch=10)
    dresults = dsampler.results


    print('######### STATIC #########')
    print(results.summary())
    print(results.eff)

    print('######### DYNAMIC #########')
    print(dresults.summary())
    print(dresults.eff)
