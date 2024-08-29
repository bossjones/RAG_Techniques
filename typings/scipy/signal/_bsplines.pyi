"""
This type stub file was generated by pyright.
"""

__all__ = ['spline_filter', 'gauss_spline', 'cspline1d', 'qspline1d', 'cspline1d_eval', 'qspline1d_eval']
def spline_filter(Iin, lmbda=...):
    """Smoothing spline (cubic) filtering of a rank-2 array.

    Filter an input data set, `Iin`, using a (cubic) smoothing spline of
    fall-off `lmbda`.

    Parameters
    ----------
    Iin : array_like
        input data set
    lmbda : float, optional
        spline smooghing fall-off value, default is `5.0`.

    Returns
    -------
    res : ndarray
        filtered input data

    Examples
    --------
    We can filter an multi dimensional signal (ex: 2D image) using cubic
    B-spline filter:

    >>> import numpy as np
    >>> from scipy.signal import spline_filter
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter = spline_filter(orig_img, lmbda=0.1)
    >>> f, ax = plt.subplots(1, 2, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...                             [sp_filter, "spline filter"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()

    """
    ...

_splinefunc_cache = ...
def gauss_spline(x, n): # -> Any:
    r"""Gaussian approximation to B-spline basis function of order n.

    Parameters
    ----------
    x : array_like
        a knot vector
    n : int
        The order of the spline. Must be non-negative, i.e., n >= 0

    Returns
    -------
    res : ndarray
        B-spline basis function values approximated by a zero-mean Gaussian
        function.

    Notes
    -----
    The B-spline basis function can be approximated well by a zero-mean
    Gaussian function with standard-deviation equal to :math:`\sigma=(n+1)/12`
    for large `n` :

    .. math::  \frac{1}{\sqrt {2\pi\sigma^2}}exp(-\frac{x^2}{2\sigma})

    References
    ----------
    .. [1] Bouma H., Vilanova A., Bescos J.O., ter Haar Romeny B.M., Gerritsen
       F.A. (2007) Fast and Accurate Gaussian Derivatives Based on B-Splines. In:
       Sgallari F., Murli A., Paragios N. (eds) Scale Space and Variational
       Methods in Computer Vision. SSVM 2007. Lecture Notes in Computer
       Science, vol 4485. Springer, Berlin, Heidelberg
    .. [2] http://folk.uio.no/inf3330/scripting/doc/python/SciPy/tutorial/old/node24.html

    Examples
    --------
    We can calculate B-Spline basis functions approximated by a gaussian
    distribution:

    >>> import numpy as np
    >>> from scipy.signal import gauss_spline
    >>> knots = np.array([-1.0, 0.0, -1.0])
    >>> gauss_spline(knots, 3)
    array([0.15418033, 0.6909883, 0.15418033])  # may vary

    """
    ...

def cspline1d(signal, lamb=...): # -> Any:
    """
    Compute cubic spline coefficients for rank-1 array.

    Find the cubic spline coefficients for a 1-D signal assuming
    mirror-symmetric boundary conditions. To obtain the signal back from the
    spline representation mirror-symmetric-convolve these coefficients with a
    length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient, default is 0.0.

    Returns
    -------
    c : ndarray
        Cubic spline coefficients.

    See Also
    --------
    cspline1d_eval : Evaluate a cubic spline at the new set of points.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a cubic spline:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import cspline1d, cspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = cspline1d_eval(cspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    ...

def qspline1d(signal, lamb=...): # -> Any:
    """Compute quadratic spline coefficients for rank-1 array.

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient (must be zero for now).

    Returns
    -------
    c : ndarray
        Quadratic spline coefficients.

    See Also
    --------
    qspline1d_eval : Evaluate a quadratic spline at the new set of points.

    Notes
    -----
    Find the quadratic spline coefficients for a 1-D signal assuming
    mirror-symmetric boundary conditions. To obtain the signal back from the
    spline representation mirror-symmetric-convolve these coefficients with a
    length 3 FIR window [1.0, 6.0, 1.0]/ 8.0 .

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a quadratic spline:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import qspline1d, qspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = qspline1d_eval(qspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    ...

def cspline1d_eval(cj, newx, dx=..., x0=...): # -> NDArray[floating[Any]]:
    """Evaluate a cubic spline at the new set of points.

    `dx` is the old sample-spacing while `x0` was the old origin. In
    other-words the old-sample points (knot-points) for which the `cj`
    represent spline coefficients were at equally-spaced points of:

      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)

    Edges are handled using mirror-symmetric boundary conditions.

    Parameters
    ----------
    cj : ndarray
        cublic spline coefficients
    newx : ndarray
        New set of points.
    dx : float, optional
        Old sample-spacing, the default value is 1.0.
    x0 : int, optional
        Old origin, the default value is 0.

    Returns
    -------
    res : ndarray
        Evaluated a cubic spline points.

    See Also
    --------
    cspline1d : Compute cubic spline coefficients for rank-1 array.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a cubic spline:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import cspline1d, cspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = cspline1d_eval(cspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    ...

def qspline1d_eval(cj, newx, dx=..., x0=...): # -> NDArray[floating[Any]]:
    """Evaluate a quadratic spline at the new set of points.

    Parameters
    ----------
    cj : ndarray
        Quadratic spline coefficients
    newx : ndarray
        New set of points.
    dx : float, optional
        Old sample-spacing, the default value is 1.0.
    x0 : int, optional
        Old origin, the default value is 0.

    Returns
    -------
    res : ndarray
        Evaluated a quadratic spline points.

    See Also
    --------
    qspline1d : Compute quadratic spline coefficients for rank-1 array.

    Notes
    -----
    `dx` is the old sample-spacing while `x0` was the old origin. In
    other-words the old-sample points (knot-points) for which the `cj`
    represent spline coefficients were at equally-spaced points of::

      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)

    Edges are handled using mirror-symmetric boundary conditions.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a quadratic spline:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import qspline1d, qspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = qspline1d_eval(qspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    ...
