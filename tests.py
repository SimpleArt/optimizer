"""
Implements test functions from
https://en.wikipedia.org/wiki/Test_functions_for_optimization

Plots the average func(x)'s each iteration.

Run main() to run the tests.
"""
from optimizer import *
import numpy as np
import matplotlib.pyplot as plt

def plot_objective(title, *args, log_scale=True, show: bool = False, learning_rate=3e-1, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("func(x)")
    (*_, x), avg_f, avg_f_slow = zip(*iter_optimize(*args, learning_rate=learning_rate, return_var=["x", "avg_func(x)", "avg_f_slow"], **kwargs))
    print(title, "Solution:", x)
    ax.plot(avg_f)
    ax.plot(avg_f_slow)
    if log_scale:
        ax.set_yscale("log")
    if show:
        plt.show()

def rastrigin(x):
    return np.mean(x**2 + 10 * (1 - np.cos(2*np.pi*x)))

def ackley(x):
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5) * np.linalg.norm(x))
        - np.exp(np.mean(np.cos(2*np.pi*x)))
        + np.e
        + 20
    )

def sphere(x):
    return np.linalg.norm(x) ** 2

def rosenbrock(x):
    return np.mean(100*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

def beale(x):
    x, y = x
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )

def goldstein_price(x):
    x, y = x
    return (
        (1 + (x+y+1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
        * (30 + (2*x-3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    )

def booth(x):
    x, y = x
    return (x + 2*y - 7) ** 2 + (2*x + y - 5) ** 2

def bukin_n6(x):
    x, y = x
    return 100 * np.sqrt(abs(y - 0.01*x**2)) + 0.01*abs(x+10)

def matyas(x):
    x, y = x
    return 0.26 * (x**2 + y**2) - 0.48*x*y

def levi_n13(x):
    x, y = x
    return (
        np.sin(3*np.pi*x) ** 2
        + (x-1) ** 2 * (1 + np.sin(3*np.pi*y) ** 2)
        + (y-1) ** 2 * (1 + np.sin(2*np.pi*y) ** 2)
    )

def himmelblau(x):
    x, y = x
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

def three_hump_camel(x):
    x, y = x
    return 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2

def easom(x):
    return -np.prod(np.cos(x)) * np.exp(-np.linalg.norm(x-np.pi)**2)

def cross_in_tray(x):
    return -1e-4 * (-holder_table(x) + 1) ** 0.1

def eggholder(x):
    x, y = x
    return -y * np.sin(np.sqrt(abs(x/2 + y))) - x * np.sin(np.sqrt(abs(x - y)))

def holder_table(x):
    return -abs(np.prod(np.sin(x))) * np.exp(abs(1 - np.linalg.norm(x) / np.pi))

def mc_cormick(x):
    x, y = x
    return np.sin(x+y) + (x-y)**2 - 1.5*x + 2.5*y + 1

def schaffer_n2(x):
    x, y = x
    return 0.5 + (np.sin(x**2 - y**2) ** 2 - 0.5) / (1 + 0.001*(x**2 + y**2)) ** 2

def schaffer_n4(x):
    x, y = x
    return 0.5 + (np.cos(np.sin(x**2 - y**2)) ** 2 - 0.5) / (1 + 0.001*(x**2 + y**2)) ** 2

def styblinski_tang(x):
    return np.mean(x**4 - 16*x**2 + 5*x)

def main():
    plot_objective("Rastrigin", rastrigin, random_array(-5.12, 5.12, 25), log_scale=False)
    plot_objective("Ackley", ackley, random_array(-5, 5, 2), log_scale=False, learning_rate=1e-1)
    plot_objective("Sphere", sphere, random_array(-5, 5, 25))
    plot_objective("Rosenbrock", rosenbrock, random_array(-2, 2, 25), learning_rate=3e-1)
    plot_objective("Beale", beale, random_array(-4.5, 4.5, 2))
    plot_objective("Goldstein-Price", goldstein_price, random_array(-2, 2, 2))
    plot_objective("Booth", booth, random_array(-2, 2, 2))
    plot_objective("Bukin function N.6", bukin_n6, random_array(-2, 2, 2), log_scale=False)
    plot_objective("Matyas", matyas, random_array(-10, 10, 2), perturbation=1e-2)
    plot_objective("Levi function N.13", levi_n13, random_array(-10, 10, 2))
    plot_objective("Himmelblau", himmelblau, random_array(-5, 5, 2), log_scale=False)
    plot_objective("3-Hump Camel", three_hump_camel, random_array(-5, 5, 2))
    plot_objective("Easom", easom, random_array(-10, 10, 2), log_scale=False, learning_rate=1e-2, perturbation=1e-2)
    plot_objective("Cross-in-Tray", cross_in_tray, random_array(-10, 10, 2), log_scale=False, abs_tol_x=1e-6, abs_tol_f=1e-6)
    plot_objective("Eggholder", eggholder, random_array(-512, 512, 2), log_scale=False)
    plot_objective("Holder Table", holder_table, random_array(-10, 10, 2), log_scale=False)
    plot_objective("McCormick", mc_cormick, random_array(-1.5, 4, 2), log_scale=False)
    plot_objective("Schaffer function N.2", schaffer_n2, random_array(-100, 100, 2), log_scale=False, learning_rate=1e-3, perturbation=3e-3)
    plot_objective("Schaffer function N.4", schaffer_n4, random_array(-100, 100, 2), log_scale=False, learning_rate=1e-2, perturbation=3e-3)
    plot_objective("Styblinski-Tang", styblinski_tang, random_array(-5, 5, 2), log_scale=False)

    plt.show()

if __name__ == "__main__":
    main()
