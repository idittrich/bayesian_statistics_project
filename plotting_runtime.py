import csv
import matplotlib.pyplot as plt
import numpy as np

dimensions = []
runtimes = []
likelihood = []
likelihood_err = []

with open('runtime.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        dimensions.append(int(row[0]))
        runtimes.append(float(row[1]))
        likelihood.append(float(row[2]))
        likelihood_err.append(float(row[4]))



fig, ax1 = plt.subplots()

ax1.plot(dimensions, runtimes, marker='o', linestyle='-', color='b')
ax1.set_xlabel('Dimensions')
ax1.set_ylabel('Elapsed Time (seconds)', color='b')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.tick_params(axis='y', labelcolor='b')

ax1.set_xlim([1, 10])

ax2 = ax1.twinx()
ax2.errorbar(dimensions, np.abs(likelihood), yerr=likelihood_err, color='r')
ax2.set_ylabel('Loglikelihood', color='r')
#ax2.set_xscale('log')
ax2.tick_params(axis='y', labelcolor='r')

ax1.grid(True)

plt.title('Runtime of Nested Sampling on the Rosenbrock Function')

plt.show()