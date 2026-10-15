__all__ = [
    'harmonica_transit_quad_ld',
    'harmonica_transit_nonlinear_ld',
    'harmonica_transit_power2_ld',
    'native',
]

import jax
import jax.numpy as jnp

from . import native
from . import custom_primitives
from .native import harmonica_transit_power2_ld


def harmonica_transit_quad_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                              u1=0., u2=0., r=jnp.array([0.1]),
                              backend='native'):
    """ Harmonica transits with jax -- quadratic limb darkening.

    See harmonica.jax.custom_primitives.harmonica_transit_quad_ld for the
    parameters. backend='native' (the default) uses the pure-jax model on
    every platform. backend='auto' uses the C++ on CPUs, which is around
    ten times faster there, but has known errors: fluxes above one for
    planets covering the stellar centre, a flattened orbit near apastron,
    and some incorrect eccentricity derivatives. See the paper on the
    paper-v2 branch for details.

    """
    return _dispatch(custom_primitives.harmonica_transit_quad_ld,
                     native.harmonica_transit_quad_ld,
                     times, t0, period, a, inc, ecc, omega, u1, u2, r=r,
                     backend=backend)


def harmonica_transit_nonlinear_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                                   u1=0., u2=0., u3=0., u4=0.,
                                   r=jnp.array([0.1]), backend='native'):
    """ Harmonica transits with jax -- non-linear limb darkening.

    See harmonica.jax.custom_primitives.harmonica_transit_nonlinear_ld for
    the parameters, and harmonica_transit_quad_ld for backend.

    """
    return _dispatch(custom_primitives.harmonica_transit_nonlinear_ld,
                     native.harmonica_transit_nonlinear_ld,
                     times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4,
                     r=r, backend=backend)


def _dispatch(cpp, pure_jax, *args, r, backend):
    """ Select an implementation once the platform is known.

    With backend='auto', the C++ evaluates each light curve point serially,
    which is fastest on a CPU, and the pure-jax model, which does a fixed
    amount of work per point, runs elsewhere. The C++ only accepts scalar
    parameters and a 1D r, so other inputs always use the pure-jax model.
    """
    if backend not in ('native', 'auto'):
        raise ValueError("backend must be 'native' or 'auto'.")
    # Copies, as jax may otherwise share the memory of NumPy inputs.
    args = [jnp.array(x, dtype=jnp.float64) for x in args]
    r = jnp.array(r, dtype=jnp.float64)
    if (backend == 'native' or r.ndim != 1
            or any(x.ndim != 0 for x in args[1:])):
        return pure_jax(*args, r=r)

    def on_cpu(*a):
        args, r = a[:-1], a[-1]
        if r.shape[0] == 1:
            return cpp(*args, r=r)
        # The C++ requires the final two coefficients not both to be zero,
        # and crashes otherwise. Such strings use the pure-jax model, and
        # the C++ is only ever given a valid stand-in, whose result is
        # discarded, in case both branches are evaluated (e.g. under vmap).
        degenerate = jnp.all(r[-2:] == 0.)
        stand_in = r.at[-2].set(jnp.where(degenerate, 1., r[-2]))
        return jax.lax.cond(degenerate,
                            lambda: pure_jax(*args, r=r),
                            lambda: cpp(*args, r=stand_in))

    return jax.lax.platform_dependent(
        *args, r, cpu=on_cpu,
        default=lambda *a: pure_jax(*a[:-1], r=a[-1]))
