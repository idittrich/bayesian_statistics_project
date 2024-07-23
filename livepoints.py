from dynesty import NestedSampler
from dynesty import plotting as dyplot
# from dynesty import utils as dyfunc
import matplotlib.pyplot as plt
import numpy as np
import time

livepoints = np.logspace(2, 5, 15, dtype=int)
dim = 3


def prior(x):
    return -1 + x * 3


def loglike(x):
    a = 1
    b = 100
    rosenbrock = sum((a - x[i])**2 + b * (x[i+1] - x[i]**2)**2 for i in range(dim - 1))
    return -rosenbrock


'''with open('livepoints.csv', 'w') as file:
    file.write("livepoints,Elapsed_Time,log(Z)\n")  # Write the header

for i, livepoint in enumerate(livepoints):
    sampler = NestedSampler(loglike, prior, dim, nlive=livepoint)
    start_time = time.time()
    sampler.run_nested(add_live=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    results = sampler.results
    print(results.summary())
    print(results.logz[-1], results.logzerr[-1])
    print(elapsed_time, " seconds")

    with open('livepoints.csv', 'a') as file:
        file.write(f"{livepoint},{elapsed_time},{results.logz[-1]},+/-,{results.logzerr[-1]}\n")'''

sampler1 = NestedSampler(loglike, prior, dim, nlive=100)
start_time = time.time()
sampler1.run_nested(add_live=False)
end_time = time.time()
elapsed_time = end_time - start_time
res1 = sampler1.results
print(res1.summary())
print(res1.logz[-1], res1.logzerr[-1])
print(elapsed_time, " seconds")

sampler2 = NestedSampler(loglike, prior, dim, nlive=10000)
start_time = time.time()

'''rlist = []
for i in range(10):
    sampler2.run_nested(add_live=False)
    rlist.append(sampler2.results)
    sampler2.reset()

end_time = time.time()
elapsed_time = end_time - start_time
res2 = dyfunc.merge_runs(rlist)'''

sampler2.run_nested(add_live=False)
end_time = time.time()
elapsed_time = end_time - start_time

res2 = sampler2.results

print(res2.summary())
print(res2.logz[-1], res2.logzerr[-1])
print(elapsed_time, " seconds")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.reshape((2, 5))

[a.set_frame_on(False) for a in axes[:, 2]]
[a.set_xticks([]) for a in axes[:, 2]]
[a.set_yticks([]) for a in axes[:, 2]]

fg, ax = dyplot.cornerpoints(res1, cmap='plasma', truths=np.zeros(dim),
                             kde=False, fig=(fig, axes[:, :2]))

fg, ax = dyplot.cornerpoints(res2, cmap='viridis', truths=np.zeros(dim),
                             kde=False, fig=(fig, axes[:, 3:]))

fig.savefig("plots/posterior_livepoints.png")


fig2, axes2 = plt.subplots(3, 7, figsize=(12, 5))
axes2 = axes2.reshape((3, 7))  # reshape axes

[a.set_frame_on(False) for a in axes2[:, 3]]
[a.set_xticks([]) for a in axes2[:, 3]]
[a.set_yticks([]) for a in axes2[:, 3]]

fg2, ax2 = dyplot.cornerplot(res1, color='green', truths=np.zeros(dim),
                             truth_color='black', show_titles=True,
                             max_n_ticks=3, quantiles=[0.025, 0.5, 0.975],
                             fig=(fig2, axes2[:, :3]))

fg2, ax2 = dyplot.cornerplot(res2, color='darkgreen', truths=np.zeros(dim),
                             truth_color='black', show_titles=True,
                             quantiles=[0.025, 0.5, 0.975], max_n_ticks=3,
                             fig=(fig2, axes2[:, 4:]))

fig2.savefig("plots/posterior_livepoints2.png")

fig3, axes3 = dyplot.traceplot(res1, truths=np.ones(dim),
                               truth_color='black', show_titles=True,
                               trace_cmap='viridis',
                               connect=False)
fig3.savefig("plots/trace1.png")

fig4, axes4 = dyplot.traceplot(res2, truths=np.ones(dim),
                               truth_color='black', show_titles=True,
                               trace_cmap='viridis',
                               connect=False)
fig4.savefig("plots/trace2.png")
