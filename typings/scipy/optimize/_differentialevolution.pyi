"""
This type stub file was generated by pyright.
"""

"""
differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
"""
__all__ = ['differential_evolution']
_MACHEPS = ...
def differential_evolution(func, bounds, args=..., strategy=..., maxiter=..., popsize=..., tol=..., mutation=..., recombination=..., seed=..., callback=..., disp=..., polish=..., init=..., atol=..., updating=..., workers=..., constraints=..., x0=..., *, integrality=..., vectorized=...): # -> OptimizeResult:
    """Finds the global minimum of a multivariate function.

    The differential evolution method [1]_ is stochastic in nature. It does
    not use gradient methods to find the minimum, and can search large areas
    of candidate space, but often requires larger numbers of function
    evaluations than conventional gradient-based techniques.

    The algorithm is due to Storn and Price [2]_.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function. The number of parameters, N, is equal
        to ``len(x)``.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. ``(min, max)`` pairs for each element in ``x``, defining the
               finite lower and upper bounds for the optimizing argument of
               `func`.

        The total number of bounds is used to determine the number of
        parameters, N. If there are parameters whose bounds are equal the total
        number of free parameters is ``N - N_equal``.

    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : {str, callable}, optional
        The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'best1exp'
            - 'rand1bin'
            - 'rand1exp'
            - 'rand2bin'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'randtobest1exp'
            - 'currenttobest1bin'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'best2bin'

        The default is 'best1bin'. Strategies that may be implemented are
        outlined in 'Notes'.
        Alternatively the differential evolution strategy can be customized by
        providing a callable that constructs a trial vector. The callable must
        have the form ``strategy(candidate: int, population: np.ndarray, rng=None)``,
        where ``candidate`` is an integer specifying which entry of the
        population is being evolved, ``population`` is an array of shape
        ``(S, N)`` containing all the population members (where S is the
        total population size), and ``rng`` is the random number generator
        being used within the solver.
        ``candidate`` will be in the range ``[0, S)``.
        ``strategy`` must return a trial vector with shape `(N,)`. The
        fitness of this trial vector is compared against the fitness of
        ``population[candidate]``.

        .. versionchanged:: 1.12.0
            Customization of evolution strategy via a callable.

    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * (N - N_equal)``
    popsize : int, optional
        A multiplier for setting the total population size. The population has
        ``popsize * (N - N_equal)`` individuals. This keyword is overridden if
        an initial population is supplied via the `init` keyword. When using
        ``init='sobol'`` the population size is calculated as the next power
        of 2 after ``popsize * (N - N_equal)``.
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Prints the evaluated `func` at every iteration.
    callback : callable, optional
        A callable called after each iteration. Has the signature:

            ``callback(intermediate_result: OptimizeResult)``

        where ``intermediate_result`` is a keyword parameter containing an
        `OptimizeResult` with attributes ``x`` and ``fun``, the best solution
        found so far and the objective function. Note that the name
        of the parameter must be ``intermediate_result`` for the callback
        to be passed an `OptimizeResult`.

        The callback also supports a signature like:

            ``callback(x, convergence: float=val)``

        ``val`` represents the fractional value of the population convergence.
        When ``val`` is greater than ``1.0``, the function halts.

        Introspection is used to determine which of the signatures is invoked.

        Global minimization will halt if the callback raises ``StopIteration``
        or returns ``True``; any polishing is still carried out.

        .. versionchanged:: 1.12.0
            callback accepts the ``intermediate_result`` keyword.

    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly. If a constrained problem is
        being studied then the `trust-constr` method is used instead. For large
        problems with many constraints, polishing can take a long time due to
        the Jacobian computations.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:

            - 'latinhypercube'
            - 'sobol'
            - 'halton'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(S, N)``, where S is the total population size and N is
              the number of parameters.
              `init` is clipped to `bounds` before use.

        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space.

        'sobol' and 'halton' are superior alternatives and maximize even more
        the parameter space. 'sobol' will enforce an initial population
        size which is calculated as the next power of 2 after
        ``popsize * (N - N_equal)``. 'halton' has no requirements but is a bit
        less efficient. See `scipy.stats.qmc` for more details.

        'random' initializes the population randomly - this has the drawback
        that clustering can occur, preventing the whole of parameter space
        being covered. Use of an array to specify a population could be used,
        for example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    updating : {'immediate', 'deferred'}, optional
        If ``'immediate'``, the best solution vector is continuously updated
        within a single generation [4]_. This can lead to faster convergence as
        trial vectors can take advantage of continuous improvements in the best
        solution.
        With ``'deferred'``, the best solution vector is updated once per
        generation. Only ``'deferred'`` is compatible with parallelization or
        vectorization, and the `workers` and `vectorized` keywords can
        over-ride this option.

        .. versionadded:: 1.2.0

    workers : int or map-like callable, optional
        If `workers` is an int the population is subdivided into `workers`
        sections and evaluated in parallel
        (uses `multiprocessing.Pool <multiprocessing>`).
        Supply -1 to use all available CPU cores.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the population in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        This option will override the `updating` keyword to
        ``updating='deferred'`` if ``workers != 1``.
        This option overrides the `vectorized` keyword if ``workers != 1``.
        Requires that `func` be pickleable.

        .. versionadded:: 1.2.0

    constraints : {NonLinearConstraint, LinearConstraint, Bounds}
        Constraints on the solver, over and above those applied by the `bounds`
        kwd. Uses the approach by Lampinen [5]_.

        .. versionadded:: 1.4.0

    x0 : None or array-like, optional
        Provides an initial guess to the minimization. Once the population has
        been initialized this vector replaces the first (best) member. This
        replacement is done even if `init` is given an initial population.
        ``x0.shape == (N,)``.

        .. versionadded:: 1.7.0

    integrality : 1-D array, optional
        For each decision variable, a boolean value indicating whether the
        decision variable is constrained to integer values. The array is
        broadcast to ``(N,)``.
        If any decision variables are constrained to be integral, they will not
        be changed during polishing.
        Only integer values lying between the lower and upper bounds are used.
        If there are no integer values lying between the bounds then a
        `ValueError` is raised.

        .. versionadded:: 1.9.0

    vectorized : bool, optional
        If ``vectorized is True``, `func` is sent an `x` array with
        ``x.shape == (N, S)``, and is expected to return an array of shape
        ``(S,)``, where `S` is the number of solution vectors to be calculated.
        If constraints are applied, each of the functions used to construct
        a `Constraint` object should accept an `x` array with
        ``x.shape == (N, S)``, and return an array of shape ``(M, S)``, where
        `M` is the number of constraint components.
        This option is an alternative to the parallelization offered by
        `workers`, and may help in optimization speed by reducing interpreter
        overhead from multiple function calls. This keyword is ignored if
        ``workers != 1``.
        This option will override the `updating` keyword to
        ``updating='deferred'``.
        See the notes section for further discussion on when to use
        ``'vectorized'``, and when to use ``'workers'``.

        .. versionadded:: 1.9.0

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully,
        ``message`` which describes the cause of the termination,
        ``population`` the solution vectors present in the population, and
        ``population_energies`` the value of the objective function for each
        entry in ``population``.
        See `OptimizeResult` for a description of other attributes. If `polish`
        was employed, and a lower minimum was obtained by the polishing, then
        OptimizeResult also contains the ``jac`` attribute.
        If the eventual solution does not satisfy the applied constraints
        ``success`` will be `False`.

    Notes
    -----
    Differential evolution is a stochastic population based method that is
    useful for global optimization problems. At each pass through the
    population the algorithm mutates each candidate solution by mixing with
    other candidate solutions to create a trial candidate. There are several
    strategies [3]_ for creating trial candidates, which suit some problems
    more than others. The 'best1bin' strategy is a good starting point for
    many systems. In this strategy two members of the population are randomly
    chosen. Their difference is used to mutate the best member (the 'best' in
    'best1bin'), :math:`x_0`, so far:

    .. math::

        b' = x_0 + mutation * (x_{r_0} - x_{r_1})

    A trial vector is then constructed. Starting with a randomly chosen ith
    parameter the trial is sequentially filled (in modulo) with parameters
    from ``b'`` or the original candidate. The choice of whether to use ``b'``
    or the original candidate is made with a binomial distribution (the 'bin'
    in 'best1bin') - a random number in [0, 1) is generated. If this number is
    less than the `recombination` constant then the parameter is loaded from
    ``b'``, otherwise it is loaded from the original candidate. The final
    parameter is always loaded from ``b'``. Once the trial candidate is built
    its fitness is assessed. If the trial is better than the original candidate
    then it takes its place. If it is also better than the best overall
    candidate it also replaces that.

    The other strategies available are outlined in Qiang and
    Mitchell (2014) [3]_.

    .. math::
            rand1* : b' = x_{r_0} + mutation*(x_{r_1} - x_{r_2})

            rand2* : b' = x_{r_0} + mutation*(x_{r_1} + x_{r_2}
                                                - x_{r_3} - x_{r_4})

            best1* : b' = x_0 + mutation*(x_{r_0} - x_{r_1})

            best2* : b' = x_0 + mutation*(x_{r_0} + x_{r_1}
                                            - x_{r_2} - x_{r_3})

            currenttobest1* : b' = x_i + mutation*(x_0 - x_i
                                                     + x_{r_0} - x_{r_1})

            randtobest1* : b' = x_{r_0} + mutation*(x_0 - x_{r_0}
                                                      + x_{r_1} - x_{r_2})

    where the integers :math:`r_0, r_1, r_2, r_3, r_4` are chosen randomly
    from the interval [0, NP) with `NP` being the total population size and
    the original candidate having index `i`. The user can fully customize the
    generation of the trial candidates by supplying a callable to ``strategy``.

    To improve your chances of finding a global minimum use higher `popsize`
    values, with higher `mutation` and (dithering), but lower `recombination`
    values. This has the effect of widening the search radius, but slowing
    convergence.

    By default the best solution vector is updated continuously within a single
    iteration (``updating='immediate'``). This is a modification [4]_ of the
    original differential evolution algorithm which can lead to faster
    convergence as trial vectors can immediately benefit from improved
    solutions. To use the original Storn and Price behaviour, updating the best
    solution once per iteration, set ``updating='deferred'``.
    The ``'deferred'`` approach is compatible with both parallelization and
    vectorization (``'workers'`` and ``'vectorized'`` keywords). These may
    improve minimization speed by using computer resources more efficiently.
    The ``'workers'`` distribute calculations over multiple processors. By
    default the Python `multiprocessing` module is used, but other approaches
    are also possible, such as the Message Passing Interface (MPI) used on
    clusters [6]_ [7]_. The overhead from these approaches (creating new
    Processes, etc) may be significant, meaning that computational speed
    doesn't necessarily scale with the number of processors used.
    Parallelization is best suited to computationally expensive objective
    functions. If the objective function is less expensive, then
    ``'vectorized'`` may aid by only calling the objective function once per
    iteration, rather than multiple times for all the population members; the
    interpreter overhead is reduced.

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Differential evolution, Wikipedia,
           http://en.wikipedia.org/wiki/Differential_evolution
    .. [2] Storn, R and Price, K, Differential Evolution - a Simple and
           Efficient Heuristic for Global Optimization over Continuous Spaces,
           Journal of Global Optimization, 1997, 11, 341 - 359.
    .. [3] Qiang, J., Mitchell, C., A Unified Differential Evolution Algorithm
            for Global Optimization, 2014, https://www.osti.gov/servlets/purl/1163659
    .. [4] Wormington, M., Panaccione, C., Matney, K. M., Bowen, D. K., -
           Characterization of structures from X-ray scattering data using
           genetic algorithms, Phil. Trans. R. Soc. Lond. A, 1999, 357,
           2827-2848
    .. [5] Lampinen, J., A constraint handling approach for the differential
           evolution algorithm. Proceedings of the 2002 Congress on
           Evolutionary Computation. CEC'02 (Cat. No. 02TH8600). Vol. 2. IEEE,
           2002.
    .. [6] https://mpi4py.readthedocs.io/en/stable/
    .. [7] https://schwimmbad.readthedocs.io/en/latest/


    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`.

    >>> import numpy as np
    >>> from scipy.optimize import rosen, differential_evolution
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = differential_evolution(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)

    Now repeat, but with parallelization.

    >>> result = differential_evolution(rosen, bounds, updating='deferred',
    ...                                 workers=2)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)

    Let's do a constrained minimization.

    >>> from scipy.optimize import LinearConstraint, Bounds

    We add the constraint that the sum of ``x[0]`` and ``x[1]`` must be less
    than or equal to 1.9.  This is a linear constraint, which may be written
    ``A @ x <= 1.9``, where ``A = array([[1, 1]])``.  This can be encoded as
    a `LinearConstraint` instance:

    >>> lc = LinearConstraint([[1, 1]], -np.inf, 1.9)

    Specify limits using a `Bounds` object.

    >>> bounds = Bounds([0., 0.], [2., 2.])
    >>> result = differential_evolution(rosen, bounds, constraints=lc,
    ...                                 seed=1)
    >>> result.x, result.fun
    (array([0.96632622, 0.93367155]), 0.0011352416852625719)

    Next find the minimum of the Ackley function
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization).

    >>> def ackley(x):
    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    >>> bounds = [(-5, 5), (-5, 5)]
    >>> result = differential_evolution(ackley, bounds, seed=1)
    >>> result.x, result.fun
    (array([0., 0.]), 4.440892098500626e-16)

    The Ackley function is written in a vectorized manner, so the
    ``'vectorized'`` keyword can be employed. Note the reduced number of
    function evaluations.

    >>> result = differential_evolution(
    ...     ackley, bounds, vectorized=True, updating='deferred', seed=1
    ... )
    >>> result.x, result.fun
    (array([0., 0.]), 4.440892098500626e-16)

    The following custom strategy function mimics 'best1bin':

    >>> def custom_strategy_fn(candidate, population, rng=None):
    ...     parameter_count = population.shape(-1)
    ...     mutation, recombination = 0.7, 0.9
    ...     trial = np.copy(population[candidate])
    ...     fill_point = rng.choice(parameter_count)
    ...
    ...     pool = np.arange(len(population))
    ...     rng.shuffle(pool)
    ...
    ...     # two unique random numbers that aren't the same, and
    ...     # aren't equal to candidate.
    ...     idxs = []
    ...     while len(idxs) < 2 and len(pool) > 0:
    ...         idx = pool[0]
    ...         pool = pool[1:]
    ...         if idx != candidate:
    ...             idxs.append(idx)
    ...
    ...     r0, r1 = idxs[:2]
    ...
    ...     bprime = (population[0] + mutation *
    ...               (population[r0] - population[r1]))
    ...
    ...     crossovers = rng.uniform(size=parameter_count)
    ...     crossovers = crossovers < recombination
    ...     crossovers[fill_point] = True
    ...     trial = np.where(crossovers, bprime, trial)
    ...     return trial

    """
    ...

class DifferentialEvolutionSolver:
    """This class implements the differential evolution solver

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function. The number of parameters, N, is equal
        to ``len(x)``.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. ``(min, max)`` pairs for each element in ``x``, defining the
               finite lower and upper bounds for the optimizing argument of
               `func`.

        The total number of bounds is used to determine the number of
        parameters, N. If there are parameters whose bounds are equal the total
        number of free parameters is ``N - N_equal``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : {str, callable}, optional
        The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'best1exp'
            - 'rand1bin'
            - 'rand1exp'
            - 'rand2bin'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'randtobest1exp'
            - 'currenttobest1bin'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'best2bin'

        The default is 'best1bin'. Strategies that may be
        implemented are outlined in 'Notes'.

        Alternatively the differential evolution strategy can be customized
        by providing a callable that constructs a trial vector. The callable
        must have the form
        ``strategy(candidate: int, population: np.ndarray, rng=None)``,
        where ``candidate`` is an integer specifying which entry of the
        population is being evolved, ``population`` is an array of shape
        ``(S, N)`` containing all the population members (where S is the
        total population size), and ``rng`` is the random number generator
        being used within the solver.
        ``candidate`` will be in the range ``[0, S)``.
        ``strategy`` must return a trial vector with shape `(N,)`. The
        fitness of this trial vector is compared against the fitness of
        ``population[candidate]``.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * (N - N_equal)``
    popsize : int, optional
        A multiplier for setting the total population size. The population has
        ``popsize * (N - N_equal)`` individuals. This keyword is overridden if
        an initial population is supplied via the `init` keyword. When using
        ``init='sobol'`` the population size is calculated as the next power
        of 2 after ``popsize * (N - N_equal)``.
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Prints the evaluated `func` at every iteration.
    callback : callable, optional
        A callable called after each iteration. Has the signature:

            ``callback(intermediate_result: OptimizeResult)``

        where ``intermediate_result`` is a keyword parameter containing an
        `OptimizeResult` with attributes ``x`` and ``fun``, the best solution
        found so far and the objective function. Note that the name
        of the parameter must be ``intermediate_result`` for the callback
        to be passed an `OptimizeResult`.

        The callback also supports a signature like:

            ``callback(x, convergence: float=val)``

        ``val`` represents the fractional value of the population convergence.
         When ``val`` is greater than ``1.0``, the function halts.

        Introspection is used to determine which of the signatures is invoked.

        Global minimization will halt if the callback raises ``StopIteration``
        or returns ``True``; any polishing is still carried out.

        .. versionchanged:: 1.12.0
            callback accepts the ``intermediate_result`` keyword.

    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly. If a constrained problem is
        being studied then the `trust-constr` method is used instead. For large
        problems with many constraints, polishing can take a long time due to
        the Jacobian computations.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:

            - 'latinhypercube'
            - 'sobol'
            - 'halton'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(S, N)``, where S is the total population size and
              N is the number of parameters.
              `init` is clipped to `bounds` before use.

        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space.

        'sobol' and 'halton' are superior alternatives and maximize even more
        the parameter space. 'sobol' will enforce an initial population
        size which is calculated as the next power of 2 after
        ``popsize * (N - N_equal)``. 'halton' has no requirements but is a bit
        less efficient. See `scipy.stats.qmc` for more details.

        'random' initializes the population randomly - this has the drawback
        that clustering can occur, preventing the whole of parameter space
        being covered. Use of an array to specify a population could be used,
        for example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    updating : {'immediate', 'deferred'}, optional
        If ``'immediate'``, the best solution vector is continuously updated
        within a single generation [4]_. This can lead to faster convergence as
        trial vectors can take advantage of continuous improvements in the best
        solution.
        With ``'deferred'``, the best solution vector is updated once per
        generation. Only ``'deferred'`` is compatible with parallelization or
        vectorization, and the `workers` and `vectorized` keywords can
        over-ride this option.
    workers : int or map-like callable, optional
        If `workers` is an int the population is subdivided into `workers`
        sections and evaluated in parallel
        (uses `multiprocessing.Pool <multiprocessing>`).
        Supply `-1` to use all cores available to the Process.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the population in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        This option will override the `updating` keyword to
        `updating='deferred'` if `workers != 1`.
        Requires that `func` be pickleable.
    constraints : {NonLinearConstraint, LinearConstraint, Bounds}
        Constraints on the solver, over and above those applied by the `bounds`
        kwd. Uses the approach by Lampinen.
    x0 : None or array-like, optional
        Provides an initial guess to the minimization. Once the population has
        been initialized this vector replaces the first (best) member. This
        replacement is done even if `init` is given an initial population.
        ``x0.shape == (N,)``.
    integrality : 1-D array, optional
        For each decision variable, a boolean value indicating whether the
        decision variable is constrained to integer values. The array is
        broadcast to ``(N,)``.
        If any decision variables are constrained to be integral, they will not
        be changed during polishing.
        Only integer values lying between the lower and upper bounds are used.
        If there are no integer values lying between the bounds then a
        `ValueError` is raised.
    vectorized : bool, optional
        If ``vectorized is True``, `func` is sent an `x` array with
        ``x.shape == (N, S)``, and is expected to return an array of shape
        ``(S,)``, where `S` is the number of solution vectors to be calculated.
        If constraints are applied, each of the functions used to construct
        a `Constraint` object should accept an `x` array with
        ``x.shape == (N, S)``, and return an array of shape ``(M, S)``, where
        `M` is the number of constraint components.
        This option is an alternative to the parallelization offered by
        `workers`, and may help in optimization speed. This keyword is
        ignored if ``workers != 1``.
        This option will override the `updating` keyword to
        ``updating='deferred'``.
    """
    _binomial = ...
    _exponential = ...
    __init_error_msg = ...
    def __init__(self, func, bounds, args=..., strategy=..., maxiter=..., popsize=..., tol=..., mutation=..., recombination=..., seed=..., maxfun=..., callback=..., disp=..., polish=..., init=..., atol=..., updating=..., workers=..., constraints=..., x0=..., *, integrality=..., vectorized=...) -> None:
        ...

    def init_population_lhs(self): # -> None:
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        ...

    def init_population_qmc(self, qmc_engine): # -> None:
        """Initializes the population with a QMC method.

        QMC methods ensures that each parameter is uniformly
        sampled over its range.

        Parameters
        ----------
        qmc_engine : str
            The QMC method to use for initialization. Can be one of
            ``latinhypercube``, ``sobol`` or ``halton``.

        """
        ...

    def init_population_random(self): # -> None:
        """
        Initializes the population at random. This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        ...

    def init_population_array(self, init): # -> None:
        """
        Initializes the population with a user specified population.

        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (S, N), where N is the number of parameters.
            The population is clipped to the lower and upper bounds.
        """
        ...

    @property
    def x(self): # -> Any:
        """
        The best solution from the solver
        """
        ...

    @property
    def convergence(self): # -> float | Any:
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        ...

    def converged(self): # -> Any | Literal[False]:
        """
        Return True if the solver has converged.
        """
        ...

    def solve(self): # -> OptimizeResult:
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully,
            ``message`` which describes the cause of the termination,
            ``population`` the solution vectors present in the population, and
            ``population_energies`` the value of the objective function for
            each entry in ``population``.
            See `OptimizeResult` for a description of other attributes. If
            `polish` was employed, and a lower minimum was obtained by the
            polishing, then OptimizeResult also contains the ``jac`` attribute.
            If the eventual solution does not satisfy the applied constraints
            ``success`` will be `False`.
        """
        ...

    def __iter__(self): # -> Self:
        ...

    def __enter__(self): # -> Self:
        ...

    def __exit__(self, *args): # -> None:
        ...

    def __next__(self): # -> tuple[Any, Any]:
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        ...



class _ConstraintWrapper:
    """Object to wrap/evaluate user defined constraints.

    Very similar in practice to `PreparedConstraint`, except that no evaluation
    of jac/hess is performed (explicit or implicit).

    If created successfully, it will contain the attributes listed below.

    Parameters
    ----------
    constraint : {`NonlinearConstraint`, `LinearConstraint`, `Bounds`}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables, shape (N,)

    Attributes
    ----------
    fun : callable
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.

    Notes
    -----
    _ConstraintWrapper.fun and _ConstraintWrapper.violation can get sent
    arrays of shape (N, S) or (N,), where S is the number of vectors of shape
    (N,) to consider constraints for.
    """
    def __init__(self, constraint, x0) -> None:
        ...

    def __call__(self, x): # -> NDArray[Any]:
        ...

    def violation(self, x): # -> NDArray[bool_]:
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables, (N, S), where N is number of
            parameters and S is the number of solutions to be investigated.

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `_ConstraintWrapper.fun`.
            Has shape (M, S) where M is the number of constraint components.
        """
        ...
