from optimizer import *
from collections import deque
from functools import partial
from numpy.polynomial import Polynomial as Poly
import numpy as np
import matplotlib.pyplot as plt

def square_norm(x):
    """A convex function with minima at the origin."""
    return np.linalg.norm(x) ** 2

print("The optimizer can be used with floats.")  # :")
print("Solution:")
print(optimize(
    square_norm,                # Minimize squared norm
    random_array(-10, 10, 25),  # in 25 dimensions from a random point
    learning_rate=0.3,          # moving roughly a third of an integer per iteration.
))
print()

print("The optimizer can be used with integers.")
print("Solution:")
print(optimize(
    square_norm,
    random_array(-10, 10, 25),
    dtype=int,
    learning_rate=0.3,
))
print()

print("Track variables as you optimize using iter_optimize(return_var=...).")
# Track the iteration, solution, and output.
for i, x, fx in iter_optimize(
    square_norm,
    random_array(-10, 10, 25, int),
    learning_rate=0.3,
    return_var=["i", "x", "func(x)"],
):
    # Only print every 200 iterations.
    if i % 200 == 0:
        print("Iteration:", i)
        print("Objective:", fx)
        print("-" * 70)
# Final iteration may not be a multiple of 200.
if i % 200 != 0:
    print("Iteration:", i)
    print("Objective:", fx)
    print("-" * 70)
print("Solution:")
print(x)
print()

print("Stochastic functions can be optimized, "
      "and common types of plots aren't very hard to make.")
# Stochastic Functions
# Functions which are noisy, meaning the output varies even if the input doesn't.
# Regression is a common example, where the target function is estimated
# inaccurately many times.

# Compute the error between the fit_with and fit_against functions on the given domain
# with the provided noise and save each point. Rather than going through the whole
# domain, a random point is uniformly chosen from it. Randomized noise is then added
# onto the fit_against function, and the squared error is returned.
def stochastic_fit(fit_with, fit_against, domain, noise, points=deque(maxlen=1)):
    def error(coefs):
        # Compute the error at a random point.
        t = random_array(*domain)
        # Add noise to the point.
        x = t + 0.1 * random_array(-noise, noise)
        y = fit_against(t) + random_array(-noise, noise)
        points.append((x, y))
        # Minimize the squared error.
        return (fit_with(coefs)(x) - y) ** 2
    return error

# Example function:
def fit_against(x):
    return np.logaddexp(0, x)

# Over the interval:
domain = [-2, 2]
# Use a Polynomial with coefficients rescaled to match the bounds:
fit_with = partial(Poly, domain=domain)
# Minimizing the stochastic error and store the last 250 points for plotting:
points = deque(maxlen=250)
func = stochastic_fit(fit_with, fit_against, domain, 0.3, points)

# Optimize using these settings:
settings = dict(
    x=[0]*5,
    learning_rate=1e-1,
    return_var=["i", "x_copy", "avg_func(x)"],
)

# Plot the fit_with, fit_against, and sampled points.
x = np.linspace(*domain)
yf = fit_against(x)
def plot_regression(i, coefs, points):
    y = fit_with(coefs)(x)
    plt.figure()
    plt.title(f"{i}th iteration")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y)
    plt.plot(x, yf, marker="*")
    plt.scatter(*zip(*points), c=(1 - 0.99 ** np.arange(len(points), 0 ,-1)).astype(str))

# Track the errors to see how the error changed over time at the end.
errors = []
for i, coefs, error in iter_optimize(func, **settings):
    errors.append(error)
    # Plot the regression every 200th iterations.
    if i % 200 == 0:
        print(f"Finished the {i}th iteration...")
        plot_regression(i, coefs, points)

# We may not terminate on a multiple of 200 iterations.
if i % 200 != 0:
    plot_regression(i, coefs, points)

# Plot the errors at the end.
plt.figure()
plt.title("Error over the iterations")
plt.xlabel("iterations")
plt.ylabel("regression error")
plt.plot(range(len(errors)), errors)

print("Solution:")
print(coefs)
plt.show()
