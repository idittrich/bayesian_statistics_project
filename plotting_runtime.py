import csv
import matplotlib.pyplot as plt

dimensions = []
runtimes = []

with open('runtime.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        dimensions.append(int(row[0]))
        runtimes.append(float(row[1]))

plt.figure(figsize=(10, 6))
plt.plot(dimensions, runtimes, marker='o', linestyle='-', color='b')
plt.xlabel('Dimensions')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Runtime of Nested Sampling on the Rosenbrock Function')
plt.grid(True)
plt.yscale('log')  # Use logarithmic scale for better visualization
plt.xscale('log')  # Use logarithmic scale for better visualization
plt.show()
