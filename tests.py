"""
Implements test functions from
https://en.wikipedia.org/wiki/Test_functions_for_optimization

Plots the average func(x)'s each iteration.

Run main() to run the tests.
"""
from optimizer import *
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

def plot_objective(title: str, *args: Any, log_scale: bool = False, show: bool = False, learning_rate: float = 3e-1, **kwargs: Any):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("y")
    (*_, x), avg_f, avg_f_slow = zip(*iter_optimize(*args, learning_rate=learning_rate, return_var=["x", "avg_func(x)", "avg_f_slow"], **kwargs))
    print(title, "Solution:", x)
    print(title, "Objective:", avg_f[-1])
    ax.plot(avg_f)
    ax.plot(avg_f_slow)
    if log_scale:
        ax.set_yscale("log")
    if show:
        plt.show()

test_cases = {}

def rastrigin(x):
    return np.mean(x**2 + 10 * (1 - np.cos(2*np.pi*x)))

test_cases["Rastrigin"] = dict(
    func=rastrigin,
    x=random_array(-5.12, 5.12, 25),
)

def ackley(x):
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5) * np.linalg.norm(x))
        - np.exp(np.mean(np.cos(2*np.pi*x)))
        + np.e
        + 20
    )

test_cases["Ackley"] = dict(
    func=ackley,
    x=random_array(-5, 5, 2),
    learning_rate=1e-1,
)

def sphere(x):
    return np.linalg.norm(x) ** 2

test_cases["Sphere"] = dict(
    func=sphere,
    x=random_array(-5, 5, 25),
    log_scale=True,
)

def rosenbrock(x):
    return np.mean(100*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

test_cases["Rosenbrock"] = dict(
    func=rosenbrock,
    x=random_array(-2, 2, 25),
    learning_rate=3e-1,
    log_scale=True,
)

def beale(x):
    x, y = x
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )

test_cases["Beale"] = dict(
    func=beale,
    x=random_array(-4.5, 4.5, 2),
    log_scale=True,
)

def goldstein_price(x):
    x, y = x
    return (
        (1 + (x+y+1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
        * (30 + (2*x-3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    )

test_cases["Goldstein-Price"] = dict(
    func=goldstein_price,
    x=random_array(-2, 2, 2),
    log_scale=True,
)

def booth(x):
    x, y = x
    return (x + 2*y - 7) ** 2 + (2*x + y - 5) ** 2

test_cases["Booth"] = dict(
    func=booth,
    x=random_array(-2, 2, 2),
    log_scale=True,
)

def bukin_n6(x):
    x, y = x
    return 100 * np.sqrt(abs(y - 0.01*x**2)) + 0.01*abs(x+10)

test_cases["Bukin function N.6"] = dict(
    func=bukin_n6,
    x=random_array(-2, 2, 2),
)

def matyas(x):
    x, y = x
    return 0.26 * (x**2 + y**2) - 0.48*x*y

test_cases["Matyas"] = dict(
    func=matyas,
    x=random_array(-10, 10, 2),
    perturbation=1e-2,
    log_scale=True,
)

def levi_n13(x):
    x, y = x
    return (
        np.sin(3*np.pi*x) ** 2
        + (x-1) ** 2 * (1 + np.sin(3*np.pi*y) ** 2)
        + (y-1) ** 2 * (1 + np.sin(2*np.pi*y) ** 2)
    )

test_cases["Levi function N.13"] = dict(
    func=levi_n13,
    x=random_array(-10, 10, 2),
)

def himmelblau(x):
    x, y = x
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

test_cases["Himmelblau"] = dict(
    func=himmelblau,
    x=random_array(-5, 5, 2),
    log_scale=True,
)

def three_hump_camel(x):
    x, y = x
    return 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2

test_cases["3-Hump Camel"] = dict(
    func=three_hump_camel,
    x=random_array(-5, 5, 2),
    log_scale=True,
)

def easom(x):
    return -np.prod(np.cos(x)) * np.exp(-np.linalg.norm(x-np.pi)**2)

test_cases["Easom"] = dict(
    func=easom,
    x=random_array(-10, 10, 2),
    learning_rate=1e-2,
    perturbation=1e-2,
    abs_tol_x=1e-3,
    abs_tol_f=1e-3,
)

def cross_in_tray(x):
    return -1e-4 * (-holder_table(x) + 1) ** 0.1

test_cases["Cross-in-Tray"] = dict(
    func=cross_in_tray,
    x=random_array(-10, 10, 2),
    abs_tol_x=1e-6,
    abs_tol_f=1e-6,
)

def eggholder(x):
    x, y = x
    return -y * np.sin(np.sqrt(abs(x/2 + y))) - x * np.sin(np.sqrt(abs(x - y)))

test_cases["Eggholder"] = dict(
    func=eggholder,
    x=random_array(-512, 512, 2),
)

def holder_table(x):
    return -abs(np.prod(np.sin(x))) * np.exp(abs(1 - np.linalg.norm(x) / np.pi))

test_cases["Holder Table"] = dict(
    func=holder_table,
    x=random_array(-10, 10, 2),
)

def mc_cormick(x):
    x, y = x
    return np.sin(x+y) + (x-y)**2 - 1.5*x + 2.5*y + 1

test_cases["McCormick"] = dict(
    func=mc_cormick,
    x=random_array(-1.5, 4, 2),
)

def schaffer_n2(x):
    x, y = x
    return 0.5 + (np.sin(x**2 - y**2) ** 2 - 0.5) / (1 + 0.001*(x**2 + y**2)) ** 2

test_cases["Schaffer function N.2"] = dict(
    func=schaffer_n2,
    x=random_array(-100, 100, 2),
    learning_rate=1e-3,
    perturbation=3e-3,
)

def schaffer_n4(x):
    x, y = x
    return 0.5 + (np.cos(np.sin(x**2 - y**2)) ** 2 - 0.5) / (1 + 0.001*(x**2 + y**2)) ** 2

test_cases["Schaffer function N.4"] = dict(
    func=schaffer_n4,
    x=random_array(-100, 100, 2),
    learning_rate=1e-2,
    perturbation=3e-3,
)

def styblinski_tang(x):
    return np.mean(x**4 - 16*x**2 + 5*x)

test_cases["Styblinski-Tang"] = dict(
    func=styblinski_tang,
    x=random_array(-5, 5, 25),
)

def main():
    for title, settings in test_cases.items():
        plot_objective(title, **settings)
    plt.show()

if __name__ == "__main__":
    main()
