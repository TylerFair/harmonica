import jax
import numpy as np
import jax.numpy as jnp
from functools import partial

from harmonica import bindings
from harmonica.jax import native

# Enable double floating precision.
jax.config.update("jax_enable_x64", True)

# Register the C++ models with jax's foreign function interface. These are
# original (api_version=0) XLA CPU custom calls.
for _name, _capsule in bindings.jax_registrations().items():
    jax.ffi.register_ffi_target(_name, _capsule, platform="cpu",
                                api_version=0)


def harmonica_transit_quad_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                              u1=0., u2=0., r=jnp.array([0.1])):
    """ Harmonica transits with jax -- quadratic limb darkening.

    Parameters
    ----------
    times : ndarray
        Array of model evaluation times [days].
    t0 : float
        Time of transit [days].
    period : float
        Orbital period [days].
    a : float
        Semi-major axis [stellar radii].
    inc : float
        Orbital inclination [radians].
    ecc : float, optional
        Eccentricity [], 0 <= ecc < 1. Default=0.
    omega : float, optional
        Argument of periastron [radians]. Default=0.
    u1, u2 : floats
        Quadratic limb-darkening coefficients.
    r : ndarray
        Transmission string coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane. The length of r must be odd,
        and the final two coefficients must not both be zero.

        .. math::

            r_{\\rm{p}}(\\theta) = \\sum_{n=0}^N a_n \\cos{(n \\theta)}
            + \\sum_{n=1}^N b_n \\sin{(n \\theta)}

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [].

    """
    params = [t0, period, a, inc, ecc, omega, u1, u2]
    return _light_curve("jax_light_curve_quad_ld", times, params, r)


def harmonica_transit_nonlinear_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                                   u1=0., u2=0., u3=0., u4=0.,
                                   r=jnp.array([0.1])):
    """ Harmonica transits with jax -- non-linear limb darkening.

    Parameters
    ----------
    times : ndarray
        Array of model evaluation times [days].
    t0 : float
        Time of transit [days].
    period : float
        Orbital period [days].
    a : float
        Semi-major axis [stellar radii].
    inc : float
        Orbital inclination [radians].
    ecc : float, optional
        Eccentricity [], 0 <= ecc < 1. Default=0.
    omega : float, optional
        Argument of periastron [radians]. Default=0.
    u1, u2, u3, u4 : floats
        Non-linear limb-darkening coefficients.
    r : ndarray
        Transmission string coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane. The length of r must be odd,
        and the final two coefficients must not both be zero.

        .. math::

            r_{\\rm{p}}(\\theta) = \\sum_{n=0}^N a_n \\cos{(n \\theta)}
            + \\sum_{n=1}^N b_n \\sin{(n \\theta)}

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [].

    """
    params = [t0, period, a, inc, ecc, omega, u1, u2, u3, u4]
    return _light_curve("jax_light_curve_nonlinear_ld", times, params, r)


def _light_curve(target, times, params, r):
    """ Check inputs, and evaluate the C++ model with its derivatives. """
    times = jnp.asarray(times, dtype=jnp.float64)
    params = [jnp.asarray(p, dtype=jnp.float64) for p in params]
    r = jnp.asarray(r, dtype=jnp.float64)
    if r.ndim != 1 or r.shape[0] % 2 != 1:
        raise ValueError('r must be a 1D array of odd length.')
    if any(p.ndim != 0 for p in params):
        # The C++ reads a single value of each parameter, so arrays would
        # be silently truncated. Use jax.vmap to batch over parameters.
        raise ValueError('Orbital and limb-darkening parameters must be '
                         'scalars; use jax.vmap to evaluate many.')
    return _flux(target, times, jnp.concatenate([jnp.stack(params), r]))


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _flux(target, times, p):
    """ C++ fluxes, as a function of the parameter vector p. """
    return _primitive(target, times, *p)[0]


@_flux.defjvp
def _flux_jvp(target, primals, tangents):
    """ First derivatives from the C++.

    The flux depends on time only through t - t0, so its time derivative
    is minus the t0 derivative.
    """
    times, p = primals
    times_dot, p_dot = tangents
    jac = _jacobian(target, times, p)
    return (_flux(target, times, p),
            jac @ p_dot - jac[..., 0] * times_dot)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _jacobian(target, times, p):
    """ C++ derivatives of the fluxes with respect to p. """
    return _primitive(target, times, *p)[1]


@_jacobian.defjvp
def _jacobian_jvp(target, primals, tangents):
    """ Second derivatives from the pure-jax model.

    The C++ computes first derivatives only. This rule is only traced for
    second and higher derivatives, so first derivatives keep the C++'s
    speed. Its own derivatives are those of the pure-jax model.
    """
    times, p = primals
    times_dot, p_dot = tangents
    n_ld = 2 if target == "jax_light_curve_quad_ld" else 4
    model = (native.harmonica_transit_quad_ld if n_ld == 2
             else native.harmonica_transit_nonlinear_ld)
    _, jac_dot = jax.jvp(
        jax.jacfwd(lambda t, q: model(t, *q[:6 + n_ld], r=q[6 + n_ld:]),
                   argnums=1),
        (times, p), (times_dot, p_dot))
    return _jacobian(target, times, p), jac_dot


def jax_light_curve_quad_ld_prim(times, *params):
    """ The C++ quadratic model, returning fluxes and derivatives.

    params are t0, period, a, inc, ecc, omega, u1, u2, then each
    coefficient of r. Only the first element of each is read.
    """
    return _primitive("jax_light_curve_quad_ld", times, *params)


def jax_light_curve_nonlinear_ld_prim(times, *params):
    """ The C++ non-linear model, returning fluxes and derivatives.

    params are t0, period, a, inc, ecc, omega, u1, u2, u3, u4, then
    each coefficient of r. Only the first element of each is read.
    """
    return _primitive("jax_light_curve_nonlinear_ld", times, *params)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _primitive(target, times, *params):
    n_rs = len(params) - (8 if target == "jax_light_curve_quad_ld" else 10)
    times = jnp.asarray(times, dtype=jnp.float64)
    params = [jnp.asarray(p, dtype=jnp.float64) for p in params]
    if n_rs > 1:
        # The C++ crashes if the final two coefficients are both zero. Raise
        # for known values; otherwise pass it a valid stand-in, and return
        # NaN, so that the call can never take the process down.
        # Like the C++, use the first element of broadcast parameters.
        degenerate = ((jnp.ravel(params[-2])[0] == 0.)
                      & (jnp.ravel(params[-1])[0] == 0.))
        if not isinstance(degenerate, jax.core.Tracer) and bool(
                jnp.any(degenerate)):
            raise ValueError(
                'The C++ model requires the final two coefficients of r not '
                'both to be zero; use harmonica.jax.native, or drop them.')
        params[-2] = jnp.where(degenerate, 1., params[-2])
        f, df_dz = _call_cpp(target, times, n_rs, params)
        return (jnp.where(degenerate, jnp.nan, f),
                jnp.where(degenerate, jnp.nan, df_dz))
    return _call_cpp(target, times, n_rs, params)


def _call_cpp(target, times, n_rs, params):
    call = jax.ffi.ffi_call(
        target,
        (jax.ShapeDtypeStruct(times.shape, jnp.float64),
         jax.ShapeDtypeStruct(times.shape + (len(params),), jnp.float64)),
        # The C++ reads one value per parameter, so batches must be
        # evaluated one call at a time.
        vmap_method="sequential",
        custom_call_api_version=2)
    # Passed as int32, which is how the C++ reads them.
    return tuple(call(np.int32(times.size), np.int32(n_rs), times, *params))


@_primitive.defjvp
def _primitive_jvp(target, primals, tangents):
    """ Tangents from the C++ derivatives; times are not differentiated.

    No tangent is given for the derivatives themselves, as they are not
    intended for gradient-based inference.
    """
    times, *params = primals
    _, *dparams = tangents
    f, df_dz = _primitive(target, times, *params)
    df = jnp.zeros_like(f)
    for idx_pd, pd in enumerate(dparams):
        df += pd * df_dz[..., idx_pd]
    return (f, df_dz), (df, jnp.zeros_like(df_dz))
