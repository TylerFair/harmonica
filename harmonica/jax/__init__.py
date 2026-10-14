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
                              u1=0., u2=0., r=jnp.array([0.1])):
    """ Harmonica transits with jax -- quadratic limb darkening.

    Uses the C++ model on CPUs and the pure-jax model elsewhere, e.g.
    on GPUs. See harmonica.jax.custom_primitives.harmonica_transit_quad_ld
    for the parameters.

    """
    return _dispatch(custom_primitives.harmonica_transit_quad_ld,
                     native.harmonica_transit_quad_ld,
                     times, t0, period, a, inc, ecc, omega, u1, u2, r=r)


def harmonica_transit_nonlinear_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                                   u1=0., u2=0., u3=0., u4=0.,
                                   r=jnp.array([0.1])):
    """ Harmonica transits with jax -- non-linear limb darkening.

    Uses the C++ model on CPUs and the pure-jax model elsewhere, e.g.
    on GPUs. See
    harmonica.jax.custom_primitives.harmonica_transit_nonlinear_ld for
    the parameters.

    """
    return _dispatch(custom_primitives.harmonica_transit_nonlinear_ld,
                     native.harmonica_transit_nonlinear_ld,
                     times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4,
                     r=r)


def _dispatch(cpp, pure_jax, *args, r):
    """ Select the faster implementation once the platform is known.

    The C++ model evaluates each light curve point serially, which is
    fastest on a CPU. The pure-jax model does a fixed amount of work per
    point, which suits GPUs. The C++ only accepts scalar parameters and
    a 1D r, so other inputs always use the pure-jax model.
    """
    args = [jnp.asarray(x, dtype=jnp.float64) for x in args]
    r = jnp.asarray(r, dtype=jnp.float64)
    if r.ndim != 1 or any(x.ndim != 0 for x in args[1:]):
        return pure_jax(*args, r=r)
    return jax.lax.platform_dependent(
        *args, r,
        cpu=lambda *a: cpp(*a[:-1], r=a[-1]),
        default=lambda *a: pure_jax(*a[:-1], r=a[-1]))
