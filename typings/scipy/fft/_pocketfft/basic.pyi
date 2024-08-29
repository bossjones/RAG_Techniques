"""
This type stub file was generated by pyright.
"""

"""
Discrete Fourier Transforms - basic.py
"""
def c2c(forward, x, n=..., axis=..., norm=..., overwrite_x=..., workers=..., *, plan=...):
    """ Return discrete Fourier transform of real or complex sequence. """
    ...

fft = ...
ifft = ...
def r2c(forward, x, n=..., axis=..., norm=..., overwrite_x=..., workers=..., *, plan=...):
    """
    Discrete Fourier transform of a real sequence.
    """
    ...

rfft = ...
ihfft = ...
def c2r(forward, x, n=..., axis=..., norm=..., overwrite_x=..., workers=..., *, plan=...):
    """
    Return inverse discrete Fourier transform of real sequence x.
    """
    ...

hfft = ...
irfft = ...
def hfft2(x, s=..., axes=..., norm=..., overwrite_x=..., workers=..., *, plan=...):
    """
    2-D discrete Fourier transform of a Hermitian sequence
    """
    ...

def ihfft2(x, s=..., axes=..., norm=..., overwrite_x=..., workers=..., *, plan=...):
    """
    2-D discrete inverse Fourier transform of a Hermitian sequence
    """
    ...

def c2cn(forward, x, s=..., axes=..., norm=..., overwrite_x=..., workers=..., *, plan=...):
    """
    Return multidimensional discrete Fourier transform.
    """
    ...

fftn = ...
ifftn = ...
def r2cn(forward, x, s=..., axes=..., norm=..., overwrite_x=..., workers=..., *, plan=...):
    """Return multidimensional discrete Fourier transform of real input"""
    ...

rfftn = ...
ihfftn = ...
def c2rn(forward, x, s=..., axes=..., norm=..., overwrite_x=..., workers=..., *, plan=...):
    """Multidimensional inverse discrete fourier transform with real output"""
    ...

hfftn = ...
irfftn = ...
def r2r_fftpack(forward, x, n=..., axis=..., norm=..., overwrite_x=...):
    """FFT of a real sequence, returning fftpack half complex format"""
    ...

rfft_fftpack = ...
irfft_fftpack = ...
