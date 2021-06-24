from __future__ import annotations
from collections import deque
from itertools import count
from typing import get_type_hints, Any, Callable, Iterable, Iterator, Optional, Sequence, Type, TypeVar, Union
from numpy.typing import ArrayLike
import numpy as np


class Momentum:
    """
    Provides an unbiased momentum i.e. an exponential moving average.

    Since it provides averages of the input values provided, no learning rates are incorporated.
    This keeps the momentum and learning rate decoupled and let's momentum be applied to other things.
    The weights are updated every iteration as well, meaning that the weight
    may be modified between updates without affecting previous estimates.
    """
    rate_minus_1: float
    momentum: ArrayLike
    weights: float
    value: ArrayLike

    def __init__(self: Momentum, *args: float, **kwargs: float) -> None:
        """
        Usage
        -----
        Momentum(0.9)
        Momentum(rate=0.9)
        Momentum(rate_minus_1=-0.1)
        """
        self.rate_minus_1 = args[0]-1 if args else kwargs.get("rate_minus_1", kwargs.get("rate", 0.999) - 1)
        self.momentum = 0.0
        self.weights = 0.0

    def __call__(self, value: ArrayLike) -> ArrayLike:
        """
        Usage
        -----
        avg_value = momentum(value)  # update and return
        avg_value == momentum.value  # retrieve the value later
        """
        self.momentum += self.rate_minus_1 * (self.momentum - value)
        self.weights += self.rate_minus_1 * (self.weights - 1)
        return self.value

    @property
    def rate(self: Momentum) -> float:
        return self.rate_minus_1 + 1

    @rate.setter
    def rate(self: Momentum, rate: float) -> None:
        self.rate_minus_1 = rate - 1

    @property
    def value(self: Momentum) -> ArrayLike:
        return self.momentum / self.weights


def random_array(
    low: ArrayLike = 0,
    high: ArrayLike = 1,
    size: Union[int, Sequence[int]] = (),
    dtype: Union[Type[float], Type[float]] = float,
) -> np.ndarray:
    """
    Produces a random array of the specified shape within the provided bounds of the specific dtype.

    Parameters
    ----------
    low : ArrayLike = 0
        The lowest possible value, inclusive.
    high : ArrayLike = 1
        The highest possible value, exclusive.
    size
        : Sequence[int] = ()
            Creates an ndarray of the given shape.
        : int
            Creates a 1-D array of the given length.
    dtype : type
        = float, default
            Create a random float value between the low/high values.
            low, high : ArrayLike[float]
        = int
            Create a random integer value between the low/high values.
            low, high : int
            Raises ValueError if np.any(low > high).

    Returns
    -------
    x : np.ndarray[dtype]
        A random numpy array.
    """
    if dtype is float:
        return np.random.uniform(low, high, size)
    elif dtype is int:
        return np.random.random_integers(low, high, size)
    else:
        raise ValueError(f"invalid dtype, requires 'float' or 'int' but got {dtype} instead.")


def optimize(
    func: Callable[[np.ndarray], float],
    x: ArrayLike,
    iterations: Optional[int] = 1000,
    abs_tol_x: float = 3e-1,
    rel_tol_x: float = 3e-1,
    abs_tol_f: float = 3e-1,
    rel_tol_f: float = 3e-1,
    refinement_iterations: int = 2500,
    avg_rate: float = 3e-2,
    minimize: bool = True,
    dtype: Union[Type[float], Type[float]] = float,
    perturbation: float = 0.5,
    pr_decay_rate: float = 0.01,
    pr_decay_power: float = 0.16,
    learning_rate: Optional[float] = None,
    lr_decay_rate: float = 0.01,
    lr_decay_power: float = 0.606,
    momentum_rate: float = 0.99,
    norm_momentum_rate: float = 0.999,
    norm_order: Optional[float] = np.inf,
    check_inputs: bool = True,
    return_var: Sequence[str] = 'x',
) -> Any:
    """
    Optimize a function i.e. find a local minima or maxima.

    The algorithm used is based on the
        Simultaneous Perturbation Stochastic Approximation (SPSA).

    SPSA is a derivative-free, global-local, stochastic algorithm that supports
    integer optimization. It may also be used as a gradient approximation algorithm.

    SPSA Features
    -------------
    Derivative-Free
        No gradient is required. Only function calls are used.
        The function need not be differentiable either for this to be used.
        For example, integer domains are possible (see below).
    Integer Optimization
        The domain of the problem may be integers instead of floats.
        Even mixed combinations of integers and floats may be used,
        see Parameters.ProblemSpecification.dtype below for more information.
    Global-Local Search
        The algorithm does not try to explore the entire search space.
        Instead, it focuses on iteratively improving the current solution.
        How far it looks to estimate the gradient depends on the perturbation
        parameter. Initially, the algorithm looks further away, and over
        time, it approaches a local search algorithm instead.
        See Parameters.HyperParameters.perturbation below for more information.
    Stochastic Optimization
        The function being optimized may be stochastic (noisy).
        If this is the case, then the expected (average) is optimized.
        Additionally, decaying hyper-parameters improves stability,
        and final results are averaged out to improve accuracy.
    Random Walk and Significant Parameters
        Parameters which matter the most experience accurate movement
        towards their optimal values. Parameters which are largely
        insignificant will mostly experience random movement, allowing
        the algorithm to search for potential changes in these areas
        without risking performance in parameters that matter.
    High Dimensionality
        Unlike other algorithms, such as Bayesian optimization, the
        dimensionality of the problem is not an issue. The algorithm
        may be used to optimize functions with hundreds of parameters.

    In addition to SPSA: Nesterov's acceleration method, momentum, and
    gradient normalization are used in combination with gradually
    decaying hyper-parameters to ensure eventual convergence.

    Other Features
    --------------
    Nesterov Acceleration
        The algorithm looks ahead and estimates the gradient at
        the next iteration instead of at the current iteration.
        This improves reactiveness to changing gradients.
    Momentum
        Various parameters are estimated using an averaging approach.
        This helps improve stability by preventing sudden changes.
        In a stochastic setting, this also makes information less noisy.
    Gradient Normalization
        The learning rate does not need to be adjusted based on the
        gradient's magnitude. Instead, the gradient is divided by a
        momentum approximation of its norm. The nature of this
        normalization is similar to that of adaptive methods, where
        we use the square root of the average squared gradient.
    Decaying Hyper Parameters
        Various hyper-parameters will decay each iteration. This
        provides theoretical convergence in the long run by allowing
        more refined iterations towards the end.
    Early Termination and Refinement
        It is usually unknown when you should terminate beforehand,
        even moreso in a stochastic setting where iterations may be noisy.
        
        See Parameters.Termination for more information.

    ==========
    Parameters
    ==========

    Required
    --------
    func(np.ndarray[dtype]) -> float
        The function being optimized. It must be able to take in numpy arrays and return floats.
        The output is either minimized or maximized depending on the `minimize` arg.
    x : ArrayLike[dtype]
        The initial estimate used. See `optimizer.random_array` to create random arrays.

    Termination
    -----------
    iterations
        : int = 1000, default
            The maximum amount of iterations ran before termination.
        : None
            The number of iterations ran is not considered for termination.
    abs_tol_x, rel_tol_x : float = 3e-1
    abs_tol_f, rel_tol_f : float = 3e-1
        The minimum default tolerance before termination.
        The x tolerance is rescaled by the learning rate.
        The relative tolerance in f is measured against a lagged average change in f.
        In other words, rel_tol_f is triggered when f starts to decrease slower.
        Set to 0 to disable.
    avg_rate : float = 3e-2
        The rate used for the averages. The smaller, the more averaged out it'll be.
        Every iteration it is gradually reduced to increase smoothness.
    refinement_iterations : int = 2500
        Run the specified number of iterations after termination.
        Note: Counts the number of iterations, x tolerance, and f tolerance separately.
              While all 3 reach their conditions, only a third of the remaining refinement iterations are used.
        Additionally applies a third of the lr/pr decay power, where the refinement_i is used for the iterations.
            learning_rate /= (1 + lr_decay_rate * refinement_i) ** (lr_decay_power / 3)
            px /= (1 + pr_decay_rate * refinement_i) ** (pr_decay_power / 3)

    Problem Specification
    ---------------------
    minimize : bool = True
        If True, `func(x)` is minimized. Otherwise `func(x)` is maximized.
    dtype : type
        = float, default
            `x` is cast to floats and the perturbation may be used.
            Hyper-parameter optimization is not usable.
        = int
            `x` is cast to integers and the perturbation is fixed to 0.5.
            If the perturbation is not set to 0.5, a warning message is prompted.
          **Hyper-parameter optimization is not usable.
        Note: To get mixed integer and float values, round the specific values yourself and set:
            dtype=float
            perturbation=0.5
            pr_decay_offset=0

    Hyper Parameters
    ----------------
    perturbation : float = 0.5
        The amount that x is changed in order to estimate the gradient.
        Changes to 0.5 if dtype=int is used for half-integer perturbations.
        Using larger perturbations reduces the amount of noise in the gradient estimates,
        but smaller perturbations increases the accuracy of the non-noise component of gradient estimates.
    pr_decay_rate : float = 0.01
    pr_decay_power : float = 0.16
        Reduce the perturbation if dtype=float according to the formula:
            perturbation / (1 + pr_decay_rate * i) ** pr_decay_power
        where `i` is the current iteration.
        This gradually improves the accuracy of the gradient estimation
        but not so fast that the noise takes over.
        Set pr_decay_...=0 to disable.
        Changes to 0 if dtype=int is used.
    learning_rate : Optional[float] = (0.1 if dtype is int else 1e-3)
        The amount that x is changed each iteration.
    lr_decay_rate : float = 0.01
    lr_decay_power : float = 0.16
        Reduce the learning_rate according to the formula:
            learning_rate / (1 + lr_decay_rate * i) ** lr_decay_power
        where `i` is the current iteration.
        This gradually improves the accuracy for stochastic functions
        and ensures eventual convergence even if the learning_rate is initially too large
        but doesn't decay so fast that it converges before the solution is found.
        Set lr_decay_offset=0 to disable.
    momentum_rate : float = 0.99
        The amount of momentum retained each iteration.
        Use 0 to disable.
    norm_momentum_rate : float = 0.999
        The amount of the magnitude of the momentum retained each iteration.
        Use 0 to disable.
    norm_order
        : float = inf, default
            The order used for the L^p norm when normalizing the momentum to compute step sizes.
        : None
            Don't normalize the momentum when computing step sizes.

    Misc.
    -----
    check_inputs : bool = True
        If True, the inputs are casted to their appropriate types and checked for appropriate conditions.
        If False, only necessary setup is done.
    return_var
        : str = 'x', default
            The name of the var returned.
            'gradient' provides the gradient estimate.
            'gradient_copy' provides a copy of the gradient estimate.
            'x_copy' provides 'x' as a copy.
            'x_float' provides 'x' without rounding.
                Note that when dtype=int, x=round_array(x_float).
            'x_float_copy' provides 'x_float' as a copy.
            'func(x)' provides the last computed func(x).
                Note that x is always perturbed during function calls,
                so this may not accurately represent func(x).
            'df(x)' provides the average change in func(x).
            'avg_...' provides an averaged estimate of the specified '...' variable, including:
                func(x)
                df(x)  (change in func(x))
                x  (not rounded)
                dx  (difference between x and avg_x)
            Any other parameters can be tracked.
        : Sequence[str]
            Returns a list of vars e.g. ['x', 'func(x)'].

    Returns
    -------
    The variable(s) specified by return_var.
    """
    return last(iter_optimize(**{k: v for k, v in locals().items() if k in get_type_hints(iter_optimize)}))


def iter_optimize(
    func: Callable[[np.ndarray], float],
    x: ArrayLike,
    iterations: Optional[int] = 1000,
    abs_tol_x: float = 3e-1,
    rel_tol_x: float = 3e-1,
    abs_tol_f: float = 3e-1,
    rel_tol_f: float = 3e-1,
    refinement_iterations: int = 2500,
    avg_rate: float = 3e-2,
    minimize: bool = True,
    dtype: Union[Type[float], Type[float]] = float,
    perturbation: float = 0.5,
    pr_decay_rate: float = 0.01,
    pr_decay_power: float = 0.16,
    learning_rate: Optional[float] = None,
    lr_decay_rate: float = 0.01,
    lr_decay_power: float = 0.606,
    momentum_rate: float = 0.99,
    norm_momentum_rate: float = 0.999,
    norm_order: Optional[float] = np.inf,
    check_inputs: bool = True,
    return_var: Sequence[str] = 'x',
) -> Iterator[Any]:
    """
    Equivalent to optimize(...) except it yields the var(s) specified by `return_var`.
    """
    def get_var(loc_vars: dict[str, Any], key: Sequence[str] = return_var) -> Any:
        if not isinstance(key, str):
            return [get_var(loc_vars, k) for k in key]
        elif key == "gradient_copy":
            return gradient.copy()
        elif key == "x_copy":
            return round_array(x) if dtype is int else x.copy()
        elif key == "x_float_copy":
            return x.copy()
        elif key == "x_float":
            return x
        elif key == "x":
            return round_array(x) if dtype is int else x
        elif key == "func(x)":
            return output
        elif key == "avg_func(x)":
            return avg_output.value
        elif key == "avg_df(x)":
            return avg_df.value
        elif key.startswith("avg_"):
            return loc_vars[key].value
        else:
            return loc_vars[key]
    # Necessary input casting.
    if learning_rate is None:
        learning_rate = 0.1 if dtype is int else 1e-3
    perturbation = abs(float(perturbation))
    if dtype is int:
        perturbation = 0.5
        pr_decay_rate = 0.0
        pr_decay_power = 0.0
    elif dtype is not float:
        raise ValueError(f"dtype expected 'float' or 'int' but got {dtype} instead.")
    x = np.array(x, copy=False, dtype=float)
    # Optional input casting.
    if check_inputs:
        if iterations is not None:
            iterations = int(iterations)
            if iterations <= 0:
                raise ValueError(f"iterations must be > 0")
        abs_tol_x = float(abs_tol_x)
        abs_tol_f = float(abs_tol_f)
        rel_tol_x = float(rel_tol_x)
        rel_tol_f = float(rel_tol_f)
        if not all(tol >= 0 for tol in (abs_tol_x, abs_tol_f, rel_tol_x, rel_tol_f)):
            raise ValueError("tolerance requires tolerance >= 0")
        perturbation = float(perturbation)
        pr_decay_rate = float(pr_decay_rate)
        pr_decay_power = float(pr_decay_power)
        learning_rate = float(learning_rate)
        lr_decay_rate = float(lr_decay_rate)
        lr_decay_power = float(lr_decay_power)
        for param in ("perturbation", "pr_decay_rate", "pr_decay_power", "learning_rate", "lr_decay_rate", "lr_decay_power"):
            if not (locals()[param] >= 0):
                raise ValueError(f"{param} requires {param} >= 0")
        if not (0 <= momentum_rate < 1):
            raise ValueError("momentum_rate requires 0 <= momentum_rate < 1.")
        if norm_order is not None:
            norm_order = float(norm_order)
    # Additional variables:
    # Define round_func to encapsulate func with the appropriate rounding.
    if dtype is float:
        round_func = func
    else:
        round_func = lambda x: func(round_array(x))
    # Track the number of iterations passed with termination.
    refinement_i = 0
    # Track the averages of several variables.
    tol_avgs = [Momentum(rate_minus_1=-avg_rate) for _ in range(4)]
    avg_x, avg_dx, avg_output, avg_df = tol_avgs
    slow_df = Momentum(rate_minus_1=-0.03)
    avg_gradient = Momentum(momentum_rate)
    gradient = np.zeros_like(x)
    avg_gradient_norm = Momentum(norm_momentum_rate)
    # Avoid division by 0.
    avg_gradient_norm(1e-7)
    # Track the last output.
    output = None
    # Define f to track the output and change to the appropriate sign.
    def f(x):
        nonlocal output
        # Track the last output.
        output = round_func(x)
        # Track the average output and change in output.
        avg_output(output)
        slow_df(avg_df(np.linalg.norm(avg_output.value - output)))
        # Correct the sign for the optimizer.
        return output if minimize else -output
    # Initialize average outputs.
    f(x)
    f(x)
    # Initial Nesterov step size is 0.
    dx = 0
    # Start from the given learning rate and decrease down to 0.
    # Then restart again from a slighly lower learning rate, but decrease slightly more.
    # Allows iterations to make big jumps and then refine the estimates.
    def restarting_decaying_learning_rate(learning_rate):
        for i in count():
            max_lr = learning_rate / (1 + lr_decay_rate * i**1.5) ** lr_decay_power
            yield from max_lr / np.sqrt(np.arange(1, 11+5*np.math.isqrt(i)))
    # Iterate over the given learning rate.
    for i, learning_rate in enumerate(restarting_decaying_learning_rate(learning_rate), 1):
        # Check if the number of iterations has been reached.
        if iterations is not None and i > iterations:
            refinement_i += 1
        # Stop if we're done refining the solution.
        if refinement_i >= refinement_iterations:
            break
        # Gradually make the averages more smooth.
        for avg in tol_avgs:
            avg.rate_minus_1 /= (1 + 1 / i) ** 0.4
        # Use a lower learning rate while refining estimates.
        learning_rate /= (1 + lr_decay_rate * refinement_i) ** (lr_decay_power / 3)
        # Generate a result.
        yield get_var(locals())
        # Compute perturbations.
        # Decay the perturbation size to gradually improve accuracy.
        px = np.random.choice([-perturbation, perturbation], x.shape)
        px /= (1 + pr_decay_rate * i) ** pr_decay_power * (1 + pr_decay_rate * refinement_i) ** (pr_decay_power / 3)
        f_plus = f(x + dx + px)
        f_minus = f(x + dx - px)
        # Estimate gradient: df/dx, where dx is the perturbation.
        df = (f_plus - f_minus) / 2
        gradient = avg_gradient(df / px)
        # Compute step size with decaying learning rate to gradually converge.
        dx = -learning_rate
        # Normalize the step size.
        if norm_order is not None:
            dx /= np.sqrt(avg_gradient_norm(np.linalg.norm(gradient, norm_order)**2))
        dx *= gradient
        x += dx
        # Check tolerances by using and updating the averages.
        if avg_dx(np.linalg.norm(dx)) < learning_rate * (abs_tol_x + rel_tol_x * np.linalg.norm(avg_x(x))):
            refinement_i += 1
        if avg_df.value < abs_tol_f + rel_tol_f * abs(slow_df.value):
            refinement_i += 1
    # Use the average solutions to reduce noise and oscillation.
    x = avg_x.value
    output = avg_output.value
    yield get_var(locals())


def round_array(x: ArrayLike) -> np.ndarray:
    """Rounds an ArrayLike to np.ndarray[int]."""
    return np.rint(x).astype(int)


T = TypeVar('T')
def last(it: Iterable[T]) -> T:
    """Exhaust and return the last item in an iterator."""
    try:
        return deque(it, maxlen=1).pop()
    except IndexError:
        raise ValueError("not enough values to unpack (expected at least 1, got 0)") from None
