""" Pure-JAX harmonica light curves, compiled by XLA for CPU or GPU.

This module evaluates the same transit model as the C++ extension, for
any number of transmission-string harmonics, without a custom call. It
can therefore be jit-compiled for GPUs, vmapped over many light curves,
and differentiated with jax.grad/jax.jacfwd, including to higher order.
Like the model itself, higher derivatives are singular at the contact
points, where the occulted area grows as a power 3/2 of the penetration.

The algorithm follows harmonica's Green's-theorem formulation, with
three numerical choices suited to fixed-shape XLA programs:

* Planet-stellar limb intersections are bracketed on a fixed angular
  grid and refined by bisection, instead of computing companion-matrix
  eigenvalues. Root derivatives are supplied by the implicit-function
  theorem. Circles use closed-form intersections, and for N_c = 1 a
  certified bracket of the two extrema resolves narrow root pairs near
  contact that a grid can miss.
* The occulted region's boundary is assembled from the direction in
  which the limbs cross, with planet arcs ordered by angle about the
  planet and stellar arcs ordered by angle about the star.
* Every planet-limb integral uses Gauss-Legendre quadrature, with closed
  forms on stellar-limb segments. The quadratic and non-linear radial
  kernels are written in cancellation-free form.
* Each light curve point executes the same fixed amount of work, padded
  to the maximum possible number of intersections.
"""
import math
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# Enable double floating precision.
jax.config.update("jax_enable_x64", True)

_PI = math.pi
_TWOPI = 2. * math.pi

# Tolerance for the trivial (no intersection) configurations.
_GEOMETRY_TOL = 1.e-12

_ROOT_BISECT_STEPS = 50


def harmonica_transit_quad_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                              u1=0., u2=0., r=jnp.array([0.1]),
                              n_legendre=50):
    """ Harmonica transits in pure jax -- quadratic limb darkening.

    A drop-in replacement for harmonica.jax.harmonica_transit_quad_ld
    which runs on any jax backend, including GPUs.

    Parameters
    ----------
    times : ndarray
        Model evaluation times [days], of any shape.
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
        of angle in the sky-plane. The length of r must be odd.

        .. math::

            r_{\\rm{p}}(\\theta) = \\sum_{n=0}^N a_n \\cos{(n \\theta)}
            + \\sum_{n=1}^N b_n \\sin{(n \\theta)}

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].
        Arrays of shape times.shape + (N,) give a string per time.
    n_legendre : int, optional
        Gauss-Legendre order of the planet-limb integrals. Default=50.

    Notes
    -----
    Orbital and limb-darkening parameters may be scalars or arrays that
    broadcast against times. To evaluate many light curves at once, e.g.
    a batch of strings, prefer adding a leading array dimension (times of
    shape (B, N) and r of shape (B, 1, 2N_c+1)) over jax.vmap: inputs are
    then evaluated in chunks of bounded memory.

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [], shaped like times.

    """
    return _light_curve('quadratic', times, (t0, period, a, inc, ecc, omega),
                        (u1, u2), r, n_legendre)


def harmonica_transit_nonlinear_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                                   u1=0., u2=0., u3=0., u4=0.,
                                   r=jnp.array([0.1]), n_legendre=50):
    """ Harmonica transits in pure jax -- non-linear limb darkening.

    A drop-in replacement for harmonica.jax.harmonica_transit_nonlinear_ld
    which runs on any jax backend, including GPUs. The parameters are as
    for harmonica_transit_quad_ld, with the four non-linear
    limb-darkening coefficients u1, u2, u3, u4.

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [], shaped like times.

    """
    return _light_curve('non-linear', times,
                        (t0, period, a, inc, ecc, omega),
                        (u1, u2, u3, u4), r, n_legendre)


def harmonica_transit_power2_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                                c=0., alpha=1., r=jnp.array([0.1]),
                                n_legendre=50):
    """ Harmonica transits in pure jax -- power-2 limb darkening.

    The stellar intensity is I(mu) = 1 - c (1 - mu^alpha). The parameters
    are otherwise as for harmonica_transit_quad_ld. Power-2 (alpha=2,
    c) is identical to quadratic (u1=2c, u2=-c).

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [], shaped like times.

    """
    return _light_curve('power-2', times, (t0, period, a, inc, ecc, omega),
                        (c, alpha), r, n_legendre)


def _light_curve(law, times, orbit_params, ld_params, r, n_legendre):
    """ Check inputs and evaluate the compiled light curve. """
    # Copy array inputs. On CPUs jax may otherwise share a NumPy array's
    # memory, and evaluation is asynchronous, so a caller modifying it
    # after this returns could change the result.
    times = jnp.array(times, dtype=jnp.float64)
    orbit_params = [jnp.array(p, dtype=jnp.float64) for p in orbit_params]
    ld_params = [jnp.array(u, dtype=jnp.float64) for u in ld_params]
    r = jnp.array(r, dtype=jnp.float64)
    if r.ndim == 0 or r.shape[-1] % 2 != 1:
        raise ValueError('The transmission string r must have an odd length.')
    # A constant zero eccentricity skips Kepler's equation entirely. A
    # traced one always takes the eccentric branch, so that derivatives
    # with respect to ecc remain correct at ecc = 0.
    ecc = orbit_params[4]
    circular_only = (not isinstance(ecc, jax.core.Tracer)
                     and bool(np.all(np.asarray(ecc) == 0.)))
    return _compiled_light_curve(law, circular_only, int(n_legendre), times,
                                 tuple(orbit_params), tuple(ld_params), r)


@partial(jax.jit, static_argnums=(0, 1, 2))
def _compiled_light_curve(law, circular_only, n_legendre, times,
                          orbit_params, ld_params, r):
    """ Broadcast inputs, solve the orbit, and evaluate each flux. """
    times = jnp.asarray(times, dtype=jnp.float64)
    orbit_params = [jnp.asarray(p, dtype=jnp.float64) for p in orbit_params]
    ld_params = [jnp.asarray(u, dtype=jnp.float64) for u in ld_params]
    shape = jnp.broadcast_shapes(times.shape, r.shape[:-1],
                                 *(p.shape for p in orbit_params + ld_params))
    d, z, nu = (jnp.broadcast_to(x, shape).ravel()
                for x in _orbit(times, *orbit_params, circular_only))

    # Map over per-time parameters only; shared ones stay unbatched.
    def batched(x, core=()):
        if x.ndim == len(core):
            return x, None
        return jnp.broadcast_to(x, shape + core).reshape((-1,) + core), 0

    (r, r_axis), *ld = (batched(r, r.shape[-1:]),
                        *(batched(u) for u in ld_params))
    axes = (0, 0, 0, r_axis) + tuple(axis for _, axis in ld)
    args = (d, z, nu, r) + tuple(u for u, _ in ld)

    def evaluate(*args):
        return jax.vmap(
            lambda d_i, z_i, nu_i, r_i, *ld_i: _flux(
                law, d_i, z_i, nu_i, r_i, ld_i, n_legendre),
            in_axes=axes)(*args)

    # Each point holds temporaries for every padded segment and quadrature
    # node. Evaluate many points in chunks, so memory stays bounded while
    # each chunk is still large enough to occupy a GPU.
    n_points = d.shape[0]
    chunk = max(256, 2 ** 22 // (_max_roots(r.shape[-1]) * n_legendre))
    if n_points <= chunk:
        return evaluate(*args).reshape(shape)
    n_chunks = -(-n_points // chunk)

    def split(x, axis):
        if axis is None:
            return x
        # Pad by repeating the last point, which is always a valid one.
        x = jnp.concatenate([x, jnp.repeat(x[-1:], n_chunks * chunk
                                           - n_points, axis=0)])
        return x.reshape((n_chunks, chunk) + x.shape[1:])

    chunked = [split(x, axis) for x, axis in zip(args, axes)]
    mapped = [i for i, axis in enumerate(axes) if axis is not None]

    def one_chunk(xs):
        full = list(chunked)
        for i, x in zip(mapped, xs):
            full[i] = x
        return evaluate(*full)

    # Checkpointing each chunk keeps reverse-mode derivatives bounded too:
    # otherwise every chunk's intermediates are stored for the backward pass.
    flux = lax.map(jax.checkpoint(one_chunk), [chunked[i] for i in mapped])
    return flux.reshape(-1)[:n_points].reshape(shape)


# ============================================================
# Orbit.
# ============================================================
def _orbit(time, t0, period, a, inc, ecc, omega, circular_only):
    """ Sky-projected separation d, line-of-sight z, and angle nu.

    Matches harmonica/orbit/trajectories.cpp, including the psi
    rotation which aligns the transmission string with the orbit.
    circular_only skips Kepler's equation, for a constant ecc = 0.
    """
    time, t0, period, a, inc, ecc, omega = jnp.broadcast_arrays(
        time, t0, period, a, inc, ecc, omega)
    n = _TWOPI / period
    sin_inc = jnp.sin(inc)
    cos_inc = jnp.cos(inc)

    def project(r, sin_fpw, cos_fpw):
        x = r * cos_fpw
        y = r * cos_inc * sin_fpw
        z = r * sin_inc * sin_fpw
        # C++ uses atan(-cos/sin), not atan2.
        psi = cos_inc * jnp.arctan(
            -cos_fpw / jnp.where(sin_fpw == 0., 1.e-300, sin_fpw))
        d = _safe_sqrt(x * x + y * y)
        nu = jnp.arctan2(y, x) - psi
        return d, z, nu

    def circular(_):
        mean_anomaly = (time - t0 + _PI / 2. / n) * n
        return project(a, jnp.sin(mean_anomaly), jnp.cos(mean_anomaly))

    def eccentric(_):
        sin_omega = jnp.sin(omega)
        cos_omega = jnp.cos(omega)
        # Eccentric anomaly at transit, where f = pi/2 - omega. Unlike the
        # half-angle form, this is regular, with correct derivatives, at
        # omega = -pi/2.
        E0 = jnp.arctan2(jnp.sqrt((1. - ecc) * (1. + ecc)) * cos_omega,
                         ecc + sin_omega)
        M0 = (1. - ecc) * E0 + ecc * _e_minus_sin(E0)
        mean_anomaly = (time - t0 + M0 / n) * n
        sinf, cosf, one_minus_ecosE = _solve_kepler(mean_anomaly, ecc)
        r = a * one_minus_ecosE
        return project(r, cosf * sin_omega + sinf * cos_omega,
                       cosf * cos_omega - sinf * sin_omega)

    return circular(None) if circular_only else eccentric(None)


def _solve_kepler(M, ecc):
    """ sin(f), cos(f) and 1 - e cos(E) from the mean anomaly M. """
    # Reduce to [-pi, pi] by a signed remainder, which keeps tiny negative
    # anomalies, and solve on [0, pi]: the other half is its mirror image.
    MA = M - _TWOPI * jnp.round(M / _TWOPI)
    MAsign = jnp.where(MA < 0., -1., 1.)
    E = _kepler_half(jnp.abs(MA), ecc)
    # Half-angle forms of 1 - e cos E and cos E - e, which do not cancel
    # as e approaches 1.
    s = jnp.sin(0.5 * E)
    sinE = MAsign * jnp.sin(E)
    denom = (1. - ecc) + 2. * ecc * s * s
    sinf = jnp.sqrt((1. - ecc) * (1. + ecc)) * sinE / denom
    cosf = ((1. - ecc) - 2. * s * s) / denom
    return sinf, cosf, denom


@jax.custom_jvp
def _kepler_half(M, ecc):
    """ Eccentric anomaly E in [0, pi] solving E - e sin E = M in [0, pi].

    Newton's method, safeguarded by bisection on the bracket [0, pi], with
    E - e sin E written as (1 - e) E + e (E - sin E) so that it does not
    cancel near periastron as e approaches 1. Derivatives are given by
    the implicit-function theorem, not through the iterations.
    """
    def g(E):
        return (1. - ecc) * E + ecc * _e_minus_sin(E) - M

    def dg(E):
        s = jnp.sin(0.5 * E)
        return (1. - ecc) + 2. * ecc * s * s

    def step(_, state):
        E, lo, hi = state
        value = g(E)
        lo = jnp.where(value < 0., E, lo)
        hi = jnp.where(value > 0., E, hi)
        newton = E - value / dg(E)
        # Keep a converged iterate, even when it lies on the bracket.
        inside = (newton >= lo) & (newton <= hi)
        E = jnp.where(value == 0., E,
                      jnp.where(inside, newton, 0.5 * (lo + hi)))
        return E, lo, hi

    E0 = jnp.clip(M + ecc * jnp.sin(M), 0., _PI)
    E, _, _ = lax.fori_loop(0, 50, step, (E0, jnp.zeros_like(M),
                                         jnp.full_like(M, _PI)))
    return E


@_kepler_half.defjvp
def _kepler_half_jvp(primals, tangents):
    M, ecc = primals
    M_t, ecc_t = tangents
    E = _kepler_half(M, ecc)
    s = jnp.sin(0.5 * E)
    dg = (1. - ecc) + 2. * ecc * s * s
    return E, (M_t + jnp.sin(E) * ecc_t) / dg


def _e_minus_sin(E):
    """ E - sin E, by its series for |E| < 0.1 to avoid cancellation. """
    small = jnp.abs(E) < 0.1
    Es = jnp.where(small, E, 0.)
    E2 = Es * Es
    series = Es * E2 * (1. / 6. - E2 * (1. / 120. - E2 * (1. / 5040. - E2 * (
        1. / 362880. - E2 * (1. / 39916800. - E2 / 6227020800.)))))
    return jnp.where(small, series, E - jnp.sin(E))


# ============================================================
# Transmission string.
# ============================================================
def _rp_drp(theta, r):
    """ Planet radius and its theta derivative, r=[a0, a1, b1, ..]. """
    return _rp_drp_cs(jnp.cos(theta), jnp.sin(theta), r)


def _rp_drp_cs(cos_t, sin_t, r):
    """ As _rp_drp, from cos(theta) and sin(theta) by angle addition. """
    rp = r[0] + jnp.zeros_like(cos_t)
    drp = jnp.zeros_like(cos_t)
    cos_nt, sin_nt = cos_t, sin_t
    for n in range(1, (r.shape[0] - 1) // 2 + 1):
        if n > 1:
            cos_nt, sin_nt = (cos_nt * cos_t - sin_nt * sin_t,
                              sin_nt * cos_t + cos_nt * sin_t)
        rp = rp + r[2 * n - 1] * cos_nt + r[2 * n] * sin_nt
        drp = drp + n * (r[2 * n] * cos_nt - r[2 * n - 1] * sin_nt)
    return rp, drp


def _geometry(theta, cos_nu, sin_nu, r):
    """ rp, drp/dtheta, cos(theta - nu) and sin(theta - nu). """
    return _geometry_cs(jnp.cos(theta), jnp.sin(theta), cos_nu, sin_nu, r)


def _geometry_cs(cos_t, sin_t, cos_nu, sin_nu, r):
    rp, drp = _rp_drp_cs(cos_t, sin_t, r)
    return (rp, drp, cos_t * cos_nu + sin_t * sin_nu,
            sin_t * cos_nu - cos_t * sin_nu)


def _intersection_f(rp, drp, cos_tmnu, sin_tmnu, d):
    """ f = rp^2 + d^2 - 2 d rp cos(theta - nu) - 1, and df/dtheta. """
    f = rp * rp + d * d - 2. * d * rp * cos_tmnu - 1.
    df = 2. * rp * drp - 2. * d * (drp * cos_tmnu - rp * sin_tmnu)
    return f, df


# ============================================================
# Planet-stellar limb intersections.
# ============================================================
def _max_roots(n_rs):
    """ f is a trigonometric polynomial of degree max(2 N_c, N_c + 1). """
    n_c = (n_rs - 1) // 2
    return 2 * max(2 * n_c, n_c + 1)


def _bisect(func, lo, hi, steps=_ROOT_BISECT_STEPS, unroll=10):
    """ Vectorised bisection of func on brackets [lo, hi]. """
    def step(_, state):
        lo, hi, flo = state
        mid = 0.5 * (lo + hi)
        fmid = func(mid)
        same = ((flo < 0.) & (fmid < 0.)) | ((flo > 0.) & (fmid > 0.))
        return (jnp.where(same, mid, lo), jnp.where(same, hi, mid),
                jnp.where(same, fmid, flo))

    lo, hi, _ = lax.fori_loop(0, steps, step, (lo, hi, func(lo)),
                              unroll=unroll)
    return 0.5 * (lo + hi)


def _sign_change_brackets(nodes, values, size):
    """ Up to size intervals [nodes_i, nodes_i+1] with a sign change.

    The nodes span a closed loop, from -pi to pi. A node at which the
    function is exactly zero takes the sign of the preceding non-zero
    node, so a crossing there is counted once, and a tangency, which
    does not change sign, is not counted at all.
    """
    values = values.at[-1].set(values[0])
    sign = jnp.sign(values)
    i = jnp.arange(values.shape[0])
    last = lax.cummax(jnp.where(sign != 0., i, -1))
    last = jnp.where(last < 0, jnp.max(jnp.where(sign != 0., i, -1)), last)
    sign = jnp.where(last < 0, 0., sign[jnp.maximum(last, 0)])
    changes = sign[:-1] * sign[1:] < 0.
    idx = jnp.nonzero(changes, size=size, fill_value=0)[0]
    # The count is not capped, so that callers can detect an overflow.
    count = jnp.sum(changes.astype(jnp.int32))
    return nodes[idx], nodes[idx + 1], jnp.arange(size) < count, count


def _grid_roots(d, nu, r, max_roots):
    """ Bracket intersections on a fixed grid and bisect each bracket.

    The grid is augmented with the bisected extrema of f, so that a
    narrow root pair inside one grid interval is still bracketed.
    """
    n_c = (r.shape[0] - 1) // 2
    grid = np.linspace(-_PI, _PI, max(128, 64 * n_c) + 1)
    cos_nu, sin_nu = jnp.cos(nu), jnp.sin(nu)
    f_grid, df_grid = _intersection_f(
        *_geometry_cs(np.cos(grid), np.sin(grid), cos_nu, sin_nu, r), d)
    f_func = lambda theta: _intersection_f(
        *_geometry(theta, cos_nu, sin_nu, r), d)[0]
    nodes, grid = jnp.asarray(grid), jnp.asarray(grid)
    if n_c >= 1:
        df_func = lambda theta: _intersection_f(
            *_geometry(theta, cos_nu, sin_nu, r), d)[1]
        lo, hi, valid, _ = _sign_change_brackets(grid, df_grid, max_roots)
        extrema = jnp.where(valid, _bisect(df_func, lo, hi), grid[0])
        nodes = jnp.sort(jnp.concatenate((grid, extrema)))
        f_grid = f_func(nodes)
    lo, hi, valid, count = _sign_change_brackets(nodes, f_grid, max_roots)
    roots = jnp.where(valid, _bisect(f_func, lo, hi), 0.)
    return roots, valid.astype(jnp.float64), count.astype(jnp.float64)


def _certified_nc1_roots(d, nu, r, max_roots):
    """ Resolve narrow N_c = 1 root pairs when the first harmonic dominates.

    Writing f as c + A cos(theta - phi) plus a second harmonic of
    amplitude B, A > sqrt(20) B guarantees exactly two stationary points.
    df/dtheta has opposite signs at phi +- pi/2 and at the endpoints of
    the complementary half circle, so bisection finds the true extrema
    even if both crossings fall inside one grid interval.
    """
    r0, a1, b1 = r[0], r[1], r[2]
    cn, sn = jnp.cos(nu), jnp.sin(nu)
    c1, s1 = 2. * r0 * (a1 - d * cn), 2. * r0 * (b1 - d * sn)
    c2 = 0.5 * (a1 * a1 - b1 * b1) - d * (a1 * cn - b1 * sn)
    s2 = a1 * b1 - d * (a1 * sn + b1 * cn)
    amplitude = jnp.hypot(c1, s1)
    certified = amplitude > jnp.sqrt(20.) * jnp.hypot(c2, s2) + 1.e-12

    def solve(_):
        phi = jnp.arctan2(s1, c1)
        left = phi + jnp.array([-0.5 * _PI, 0.5 * _PI])

        def df_func(theta):
            return (-amplitude * jnp.sin(theta - phi)
                    - 2. * c2 * jnp.sin(2. * theta)
                    + 2. * s2 * jnp.cos(2. * theta))

        f_func = lambda theta: _intersection_f(
            *_geometry(theta, cn, sn, r), d)[0]
        # Two brackets each: unrolling only adds overhead here.
        extrema = _bisect(df_func, left, left + _PI, unroll=1)
        values = f_func(extrema)
        partial = (values[0] > 0.) & (values[1] < 0.)
        pair = _bisect(f_func, extrema,
                       jnp.array([extrema[1], extrema[0] + _TWOPI]), unroll=1)
        pair = jnp.sort(jnp.mod(pair + _PI, _TWOPI) - _PI)
        roots = jnp.zeros(max_roots).at[:2].set(jnp.where(partial, pair, 0.))
        valid = jnp.zeros(max_roots).at[:2].set(jnp.where(partial, 1., 0.))
        return roots, valid, jnp.where(partial, 2., 0.)

    return lax.cond(certified, solve,
                    lambda _: _grid_roots(d, nu, r, max_roots), None)


@jax.custom_jvp
def _intersections(d, nu, r):
    """ Sorted intersections in [-pi, pi], padded to max(4 N_c, 2).

    Returns (thetas, valid, count). Bracketing is discrete, so the
    returned root tangents are given by the implicit-function theorem.
    """
    max_roots = _max_roots(r.shape[0])

    def circle(_):
        # Factored discriminant, accurate at contact.
        r0 = r[0]
        partial = (d > jnp.abs(1. - r0)) & (d < 1. + r0) & (r0 > 0.)
        disc = (1. - (d - r0) ** 2) * ((d + r0) ** 2 - 1.)
        angle = jnp.arctan2(jnp.sqrt(jnp.maximum(disc, 0.)),
                            d * d + r0 * r0 - 1.)
        pair = jnp.sort(jnp.mod(jnp.array([nu - angle, nu + angle]) + _PI,
                                _TWOPI) - _PI)
        roots = jnp.zeros(max_roots).at[:2].set(jnp.where(partial, pair, 0.))
        valid = jnp.zeros(max_roots).at[:2].set(jnp.where(partial, 1., 0.))
        return roots, valid, jnp.where(partial, 2., 0.)

    if r.shape[0] == 1:
        return circle(None)
    general = _certified_nc1_roots if r.shape[0] == 3 else _grid_roots
    return lax.cond(jnp.all(r[1:] == 0.), circle,
                    lambda _: general(d, nu, r, max_roots), None)


@_intersections.defjvp
def _intersections_jvp(primals, tangents):
    d, nu, r = primals
    d_t, nu_t, r_t = tangents
    roots, valid, count = _intersections(d, nu, r)

    cos_t, sin_t = jnp.cos(roots), jnp.sin(roots)
    rp, drp, cos_tmnu, sin_tmnu = _geometry_cs(cos_t, sin_t, jnp.cos(nu),
                                               jnp.sin(nu), r)
    rp_t, _ = _rp_drp_cs(cos_t, sin_t, r_t)
    _, df_dtheta = _intersection_f(rp, drp, cos_tmnu, sin_tmnu, d)
    df_t = (2. * (rp - d * cos_tmnu) * rp_t
            + 2. * (d - rp * cos_tmnu) * d_t
            - 2. * d * rp * sin_tmnu * nu_t)
    # Padded roots can have df/dtheta = 0; guard before dividing so that
    # reverse mode cannot leak 0/0.
    nonsingular = (valid > 0.5) & (jnp.abs(df_dtheta) > 1.e-14)
    safe = jnp.where(nonsingular, df_dtheta, 1.)
    roots_t = jnp.where(nonsingular, -df_t / safe, 0.)
    return (roots, valid, count), (roots_t, jnp.zeros_like(valid),
                                   jnp.zeros_like(count))


# ============================================================
# Limb-darkening laws.
# ============================================================
def _law(law, ld):
    """ Polynomial coefficients p_n, and the normalisation I_0. """
    if law == 'quadratic':
        u1, u2 = ld
        p = jnp.stack([1. - u1 - u2, u1 + 2. * u2, -u2])
        bt = 1. - u1 / 3. - u2 / 6.
    elif law == 'non-linear':
        u1, u2, u3, u4 = ld
        p = jnp.stack([1. - u1 - u2 - u3 - u4, u1, u2, u3, u4])
        bt = 1. - u1 / 5. - u2 / 3. - 3. * u3 / 7. - u4 / 2.
    else:
        c, alpha = ld
        p = jnp.stack([1. - c, c])
        bt = 1. - c + 2. * c / (alpha + 2.)
    return p, 1. / (_PI * bt)


def _star_zetas(law, ld):
    """ Radial kernels on the stellar limb (zp=0), 1 / (n + 2). """
    if law == 'quadratic':
        return jnp.array([1. / 2., 1. / 3., 1. / 4.])
    if law == 'non-linear':
        return jnp.array([1. / 2., 2. / 5., 1. / 3., 2. / 7., 1. / 4.])
    return jnp.stack([jnp.asarray(0.5), 1. / (ld[1] + 2.)])


def _planet_zetas(law, ld, zp, zp_sq):
    """ zeta_n = (1 - zp^(n+2)) / ((n+2) (1 - zp^2)) for each LD term.

    Written without the removable singularity at zp = 1, the stellar
    disc centre, and the cancellation near it. zeta_0 = 1/2 is returned
    as a scalar, so that its integral reduces to a plain sum.
    """
    zeta_0 = 0.5
    zeta_1 = (zp + 1. / (1. + zp)) / 3.
    zeta_2 = 0.25 * (1. + zp_sq)
    if law == 'quadratic':
        return zeta_0, zeta_1, zeta_2
    if law == 'non-linear':
        # With s = sqrt(zp): (1 - s^k) / (1 - s^4) as finite sums.
        s = _safe_sqrt(zp)
        s2 = s * s
        denom = (1. + s) * (1. + s2)
        geometric_5 = 1. + s + s2 + s2 * s + s2 * s2
        geometric_7 = geometric_5 + s2 * s2 * s + s2 * s2 * s2
        return (zeta_0, 0.4 * geometric_5 / denom, zeta_1,
                geometric_7 / (3.5 * denom), zeta_2)
    return zeta_0, _power_zeta(ld[1] + 2., zp, zp_sq)


def _power_zeta(k, zp, zp_sq):
    """ (1 - zp^k) / (k (1 - zp^2)) for real k > 0, stable to zp = 1.

    With t = 1 - zp^2 and p = k / 2, this is (1 - (1 - t)^p) / (k t).
    """
    t = 1. - zp_sq
    p = 0.5 * k
    # A Taylor series near the disc centre, whose low-order derivatives
    # are accurate too. Keeping p t <= 0.1 bounds its truncation error
    # by ~1e-21 for any exponent.
    series = t < jnp.minimum(0.01, 0.1 / p)
    near = t < 0.5
    t_series = jnp.where(series, t, 0.)
    term, total = p, p
    for j in range(1, 12):
        term = -term * (p - j) / (j + 1.) * t_series
        total = total + term
    zeta_series = total / k
    # 1 - (1 - t)^p = -expm1(p log1p(-t)) further out.
    t_near = jnp.where(near & ~series, t, 0.25)
    zeta_near = -jnp.expm1(p * jnp.log1p(-t_near)) / (k * t_near)
    zp_far = jnp.where(near, 0.5, zp)
    t_far = jnp.where(near, 0.75, t)
    zeta_far = (1. - jnp.power(zp_far, k)) / (k * t_far)
    return jnp.where(series, zeta_series,
                     jnp.where(near, zeta_near, zeta_far))


# ============================================================
# Line integrals.
# ============================================================
def _planet_segment(law, ld, theta_j, theta_jp1, d, nu, r, roots, weights,
                    active):
    """ Gauss-Legendre s_n integrals along the planet's limb. """
    # Inactive padded segments may sit on a singular point; replace their
    # geometry so that masked-out values cannot leak NaN derivatives.
    theta_j = jnp.where(active, theta_j, 0.)
    theta_jp1 = jnp.where(active, theta_jp1, 0.)
    half_range = 0.5 * (theta_jp1 - theta_j)
    t_k = half_range * (roots + 1.) + theta_j
    rp, drp, cos_tmnu, sin_tmnu = _geometry(t_k, jnp.cos(nu), jnp.sin(nu),
                                            r)
    d_rp_cos = d * rp * cos_tmnu
    eta = rp * rp - d_rp_cos - d * drp * sin_tmnu
    zp_sq = jnp.maximum(1. - d * d - rp * rp + 2. * d_rp_cos, 0.)
    eta = jnp.where(active, eta, 0.)
    zp_sq = jnp.where(active, zp_sq, 0.25)
    zetas = _planet_zetas(law, ld, _safe_sqrt(zp_sq), zp_sq)
    # Separate reductions fuse with their integrands; a stacked
    # (n_terms, n_legendre) product measured ~25% slower.
    weighted_eta = eta * weights
    return half_range * jnp.stack([jnp.sum(zeta * weighted_eta)
                                   for zeta in zetas])


def _limb_phi(theta, d, nu, r, active):
    """ Angle about the stellar centre of the planet-limb point theta.

    Inactive entries may coincide with the stellar centre, where arctan2
    has no derivative; they are replaced before it is called.
    """
    rp, _, cos_tmnu, sin_tmnu = _geometry(theta, jnp.cos(nu), jnp.sin(nu), r)
    return jnp.arctan2(jnp.where(active, -rp * sin_tmnu, 0.),
                       jnp.where(active, -rp * cos_tmnu + d, 1.))


def _flux(law, d, z, nu, r, ld, n_legendre):
    """ Normalised flux for one sky position. """
    p, I_0 = _law(law, ld)
    s = _occulted_integrals(law, ld, d, z, nu, r, n_legendre)
    return 1. - I_0 * jnp.sum(p * s)


def _occulted_integrals(law, ld, d, z, nu, r, n_legendre):
    """ The s_n vector integrated around the occulted region. """
    max_roots = _max_roots(r.shape[0])
    gl_roots, gl_weights = (jnp.asarray(x) for x in
                            np.polynomial.legendre.leggauss(n_legendre))
    an_bn = jnp.hypot(r[1::2], r[2::2])
    # Exact extremes for N_c <= 1, otherwise conservative bounds.
    min_rp = r[0] - jnp.sum(an_bn)
    max_rp = r[0] + jnp.sum(an_bn)
    inside = d <= 1.
    triv_entire_planet = inside & (max_rp <= 1. - d + _GEOMETRY_TOL)
    triv_entire_star = min_rp >= 1. + d - _GEOMETRY_TOL
    triv_beyond = ~inside & (max_rp <= d - 1. + _GEOMETRY_TOL)
    trivial = triv_entire_planet | triv_entire_star | triv_beyond

    thetas, _, count = _intersections(d, nu, r)
    count = jnp.rint(count).astype(jnp.int32)
    # Tangencies are not counted, so a closed curve crosses an even number
    # of times. An odd count, or more crossings than f can have, means the
    # bracketing failed, and the flux is returned as NaN rather than as a
    # plausible but wrong value. A tolerance-classified trivial geometry
    # may still have a root pair within tolerance of contact; it must
    # integrate the entire limb.
    unresolved = (count % 2 == 1) | (count > max_roots)
    has_roots = (count >= 2) & ~unresolved & ~trivial

    # Planet-limb segments [theta_j, theta_j+1], closed by theta_0 + 2 pi.
    slot = jnp.arange(max_roots)
    in_use = slot < count
    padded = jnp.where(in_use, thetas, thetas[0])
    ends = jnp.where(in_use, jnp.where(slot == count - 1,
                                       padded[0] + _TWOPI,
                                       jnp.roll(padded, -1)), padded)
    # Without roots, the first slot integrates the planet's entire limb.
    first = slot == 0
    starts = jnp.where(has_roots, padded, jnp.where(first, nu - _PI, 0.))
    ends = jnp.where(has_roots, ends, jnp.where(first, nu + _PI, 0.))

    # The occulted region's boundary follows from the direction of each
    # crossing. f < 0 inside the star, so the planet arc after a crossing
    # bounds the region when f_theta < 0 there. Ordered instead by angle
    # phi about the stellar centre, the stellar arc after a crossing bounds
    # it when f_theta > 0, because the planet's inside function on the
    # stellar limb has derivative f_theta at a crossing. Unlike a lookup
    # of each crossing's type, this is unaffected by tangencies, and each
    # stellar arc runs anticlockwise.
    _, f_theta = _intersection_f(
        *_geometry(padded, jnp.cos(nu), jnp.sin(nu), r), d)
    planet_arc = jnp.where(has_roots, in_use & (f_theta < 0.), first)
    phi = _limb_phi(padded, d, nu, r, in_use & has_roots)
    order = jnp.argsort(jnp.where(in_use, phi, jnp.inf))
    phi_start = phi[order]
    phi_end = jnp.where(slot == count - 1, phi_start[0] + _TWOPI,
                        jnp.roll(phi_start, -1))
    star_arc = has_roots & in_use & (f_theta[order] > 0.)
    star = jnp.where(star_arc, phi_end - phi_start, 0.)

    planet = jax.vmap(lambda theta_j, theta_jp1, is_active: _planet_segment(
        law, ld, theta_j, theta_jp1, d, nu, r, gl_roots, gl_weights,
        is_active))(starts, ends, planet_arc)
    intersecting = (jnp.sum(planet, axis=0)
                    + jnp.sum(star) * _star_zetas(law, ld))

    entire_planet = planet[0]
    entire_star = _TWOPI * _star_zetas(law, ld)
    zeros = jnp.zeros_like(entire_star)
    # Without intersections, the limb point facing the stellar centre
    # decides between the remaining configurations, as in the C++.
    rp_nu, _ = _rp_drp(nu, r)
    no_roots = jnp.where(rp_nu >= 1. + d, entire_star,
                         jnp.where(inside, entire_planet, zeros))
    s = jnp.where(triv_entire_planet, entire_planet,
                  jnp.where(triv_entire_star, entire_star,
                            jnp.where(triv_beyond, zeros,
                                      jnp.where(has_roots, intersecting,
                                                no_roots))))
    s = jnp.where(unresolved & ~trivial, jnp.nan, s)
    return jnp.where(z > 0., s, zeros)


@jax.custom_jvp
def _safe_sqrt(x):
    """ sqrt(max(x, 0)), with a zero derivative at and below zero. """
    return jnp.sqrt(jnp.maximum(x, 0.))


@_safe_sqrt.defjvp
def _safe_sqrt_jvp(primals, tangents):
    (x,), (x_t,) = primals, tangents
    value = _safe_sqrt(x)
    return value, jnp.where(x > 0., 0.5 * x_t / jnp.where(x > 0., value, 1.),
                            0.)
