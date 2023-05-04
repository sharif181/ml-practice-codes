import numpy as np
import matplotlib.pyplot as plt

val = np.array([10, 12, 10, 9, 13, 2, 10, 10, 10])

print(np.percentile(val, 25))


def normal_dist(x, mean, sd):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density


# Calculate mean and Standard deviation.
mean = np.mean(val)
sd = np.std(val)

pdf = normal_dist(val, mean, sd)

# Plotting the Results
plt.plot(val, pdf, color='red')
plt.xlabel('Data points')
plt.ylabel('Probability Density')
plt.show()
