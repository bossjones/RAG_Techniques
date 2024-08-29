"""
This type stub file was generated by pyright.
"""

"""Compute the action of the matrix exponential."""
__all__ = ['expm_multiply']
def traceest(A, m3, seed=...):
    """Estimate `np.trace(A)` using `3*m3` matrix-vector products.

    The result is not deterministic.

    Parameters
    ----------
    A : LinearOperator
        Linear operator whose trace will be estimated. Has to be square.
    m3 : int
        Number of matrix-vector products divided by 3 used to estimate the
        trace.
    seed : optional
        Seed for `numpy.random.default_rng`.
        Can be provided to obtain deterministic results.

    Returns
    -------
    trace : LinearOperator.dtype
        Estimate of the trace

    Notes
    -----
    This is the Hutch++ algorithm given in [1]_.

    References
    ----------
    .. [1] Meyer, Raphael A., Cameron Musco, Christopher Musco, and David P.
       Woodruff. "Hutch++: Optimal Stochastic Trace Estimation." In Symposium
       on Simplicity in Algorithms (SOSA), pp. 142-155. Society for Industrial
       and Applied Mathematics, 2021
       https://doi.org/10.1137/1.9781611976496.16

    """
    ...

def expm_multiply(A, B, start=..., stop=..., num=..., endpoint=..., traceA=...):
    """
    Compute the action of the matrix exponential of A on B.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix or vector to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    endpoint : bool, optional
        If True, `stop` is the last time point.  Otherwise, it is not included.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable.
        For linear operators, `traceA` should be provided to ensure performance
        as the estimation is not guaranteed to be reliable for all cases.

        .. versionadded:: 1.9.0

    Returns
    -------
    expm_A_B : ndarray
         The result of the action :math:`e^{t_k A} B`.

    Warns
    -----
    UserWarning
        If `A` is a linear operator and ``traceA=None`` (default).

    Notes
    -----
    The optional arguments defining the sequence of evenly spaced time points
    are compatible with the arguments of `numpy.linspace`.

    The output ndarray shape is somewhat complicated so I explain it here.
    The ndim of the output could be either 1, 2, or 3.
    It would be 1 if you are computing the expm action on a single vector
    at a single time point.
    It would be 2 if you are computing the expm action on a vector
    at multiple time points, or if you are computing the expm action
    on a matrix at a single time point.
    It would be 3 if you want the action on a matrix with multiple
    columns at multiple time points.
    If multiple time points are requested, expm_A_B[0] will always
    be the action of the expm at the first time point,
    regardless of whether the action is on a vector or a matrix.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
           "Computing the Action of the Matrix Exponential,
           with an Application to Exponential Integrators."
           SIAM Journal on Scientific Computing,
           33 (2). pp. 488-511. ISSN 1064-8275
           http://eprints.ma.man.ac.uk/1591/

    .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
           "Computing Matrix Functions."
           Acta Numerica,
           19. 159-208. ISSN 0962-4929
           http://eprints.ma.man.ac.uk/1451/

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import expm, expm_multiply
    >>> A = csc_matrix([[1, 0], [0, 1]])
    >>> A.toarray()
    array([[1, 0],
           [0, 1]], dtype=int64)
    >>> B = np.array([np.exp(-1.), np.exp(-2.)])
    >>> B
    array([ 0.36787944,  0.13533528])
    >>> expm_multiply(A, B, start=1, stop=2, num=3, endpoint=True)
    array([[ 1.        ,  0.36787944],
           [ 1.64872127,  0.60653066],
           [ 2.71828183,  1.        ]])
    >>> expm(A).dot(B)                  # Verify 1st timestep
    array([ 1.        ,  0.36787944])
    >>> expm(1.5*A).dot(B)              # Verify 2nd timestep
    array([ 1.64872127,  0.60653066])
    >>> expm(2*A).dot(B)                # Verify 3rd timestep
    array([ 2.71828183,  1.        ])
    """
    ...

_theta = ...
class LazyOperatorNormInfo:
    """
    Information about an operator is lazily computed.

    The information includes the exact 1-norm of the operator,
    in addition to estimates of 1-norms of powers of the operator.
    This uses the notation of Computing the Action (2011).
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """
    def __init__(self, A, A_1_norm=..., ell=..., scale=...) -> None:
        """
        Provide the operator and some norm-related information.

        Parameters
        ----------
        A : linear operator
            The operator of interest.
        A_1_norm : float, optional
            The exact 1-norm of A.
        ell : int, optional
            A technical parameter controlling norm estimation quality.
        scale : int, optional
            If specified, return the norms of scale*A instead of A.

        """
        ...

    def set_scale(self, scale): # -> None:
        """
        Set the scale parameter.
        """
        ...

    def onenorm(self): # -> floating[Any]:
        """
        Compute the exact 1-norm.
        """
        ...

    def d(self, p):
        """
        Lazily estimate :math:`d_p(A) ~= || A^p ||^(1/p)` where :math:`||.||` is the 1-norm.
        """
        ...

    def alpha(self, p):
        """
        Lazily compute max(d(p), d(p+1)).
        """
        ...
