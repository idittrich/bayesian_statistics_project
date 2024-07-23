import csv
import matplotlib.pyplot as plt
import numpy as np

livepoints = []
runtimes = []
likelihood = []
likelihood_err = []

with open('livepoints.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        livepoints.append(int(row[0]))
        runtimes.append(float(row[1]))
        likelihood.append(float(row[2]))
        likelihood_err.append(float(row[4]))

livepoints = np.array(livepoints)
runtimes = np.array(runtimes)
likelihood = np.array(likelihood)
likelihood_err = np.array(likelihood_err)

likelihood_mean = np.mean(likelihood)
likelihood_mean_err = np.sqrt(np.sum((likelihood_err/len(likelihood_err))**2))
print("likelihood mean: ", likelihood_mean, likelihood_mean_err)

coefficients_linear, covariance_matrix = np.polyfit(np.log10(livepoints),
                                                    np.log10(runtimes),
                                                    1,
                                                    cov=True)

polynomial_linear = np.poly1d(coefficients_linear)

standard_errors = np.sqrt(np.diag(covariance_matrix))
print("linear fit", coefficients_linear, standard_errors)

x_values = np.logspace(np.log10(livepoints.min()), np.log10(livepoints.max()), 100)

fit_values = 10 ** polynomial_linear(np.log10(x_values))

coeffs_upper = coefficients_linear + standard_errors
coeffs_lower = coefficients_linear - standard_errors

fit_upper = 10 ** (np.polyval(coeffs_upper, np.log10(x_values)))
fit_lower = 10 ** (np.polyval(coeffs_lower, np.log10(x_values)))

plt.figure(figsize=(7, 5))
plt.grid(True, which="both", ls="--")
plt.fill_between(x_values, fit_lower, fit_upper, color='green', alpha=0.3, label=r'$\sigma$')
plt.plot(x_values, fit_values, color='green', label='Linear Fit ($O(n_{live}^k$), $k=0.98\pm0.02$)', linestyle='--')
plt.plot(livepoints, livepoints/180, color='darkgreen', label=r'$O(n_{live})$')
plt.scatter(livepoints, runtimes, color='black', label='Runtime Data', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\log(n_{live})$', fontsize=16)
plt.ylabel(r'$\log$(runtime) [s]', fontsize=16)
plt.legend(fontsize=14)
plt.savefig('plots/runtime_livepoints_plot.png')

plt.figure(figsize=(7, 5))
plt.errorbar(livepoints, likelihood, likelihood_err, 0, capsize=5,
             elinewidth=1, capthick=2, ecolor='black', color='black', label=r'$\log(Z)$ measurement')
plt.axhline(y=likelihood_mean, color='green', linestyle='--', label=r'$\langle\log(Z)\rangle=-3.70\pm0.02$')
plt.fill_between(livepoints, likelihood_mean+likelihood_mean_err, likelihood_mean-likelihood_mean_err, color='green', alpha=0.3)
plt.xscale('log')
plt.grid(color='grey')
plt.xlabel(r'$\log(n_{live})$', fontsize=16)
plt.ylabel(r'$\log$(Z)', fontsize=16)
plt.legend()
plt.savefig('plots/likelihood_livepoints_plot.png')
