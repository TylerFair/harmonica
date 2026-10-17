import unittest
import numpy as np
import jax
import jax.numpy as jnp

from harmonica import HarmonicaTransit
from harmonica.jax import native


def _brute_force_area(d, nu, r, n=2000000):
    """ Overlap of a unit disc with the planet, integrated radially. """
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    rp = np.full_like(theta, r[0])
    for k in range(1, (len(r) - 1) // 2 + 1):
        rp += r[2 * k - 1] * np.cos(k * theta) + r[2 * k] * np.sin(k * theta)
    c = d * np.cos(theta - nu)
    disc = c * c - d * d + 1.
    root = np.sqrt(np.maximum(disc, 0.))
    lo, hi = np.maximum(c - root, 0.), np.minimum(c + root, rp)
    inside = (disc > 0.) & (hi > lo)
    return np.sum(np.where(inside, 0.5 * (hi * hi - lo * lo), 0.)) * 2. * np.pi / n


def _uniform_flux(d, nu, r):
    """ Native flux of a uniform star at one sky position. """
    zero = jnp.asarray(0.)
    s = native._occulted_integrals('quadratic', (zero, zero), jnp.asarray(d),
                                   jnp.asarray(1.), jnp.asarray(nu),
                                   jnp.asarray(r, dtype=jnp.float64), 200)
    return 1. - float(s[0]) / np.pi


class TestJaxNative(unittest.TestCase):
    """ Test the pure-jax light curves against the C++ implementation. """

    def __init__(self, *args, **kwargs):
        super(TestJaxNative, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(5)

        # Example params.
        self.t0 = 5.
        self.period = 10.
        self.a = 10.
        self.inc = 89. * np.pi / 180.
        self.times = np.linspace(4.7, 5.3, 1000)
        self.quad_ld = np.array([0.1, 0.5])
        self.nl_ld = np.array([0.1, 0.2, 0.3, 0.4])

    def _random_string(self, n_c, r0=0.1):
        rs = np.zeros(2 * n_c + 1)
        rs[0] = r0
        rs[1:] = np.random.normal(0., 0.1 * r0 / max(n_c, 1), 2 * n_c)
        return rs

    def _cpp(self, ecc, omega, us, rs, limb_dark_law):
        ht = HarmonicaTransit(times=self.times, pnl_c=500, pnl_e=500)
        ht.set_orbit(self.t0, self.period, self.a, self.inc, ecc, omega)
        ht.set_stellar_limb_darkening(us, limb_dark_law=limb_dark_law)
        ht.set_planet_transmission_string(rs)
        return ht.get_transit_light_curve()

    def _jax(self, ecc, omega, us, rs, limb_dark_law):
        if limb_dark_law == 'quadratic':
            fn = native.harmonica_transit_quad_ld
        else:
            fn = native.harmonica_transit_nonlinear_ld
        return np.asarray(fn(self.times, self.t0, self.period, self.a,
                             self.inc, ecc, omega, *us, r=rs))

    def test_cpp_parity(self):
        """ Test fluxes match the C++ for each law, orbit and N_c. """
        for limb_dark_law, us in zip(['quadratic', 'non-linear'],
                                     [self.quad_ld, self.nl_ld]):
            for ecc, omega in zip([0., 0.1], [0., 0.1 * np.pi / 180.]):
                for n_c in range(4):
                    rs = self._random_string(n_c)
                    fs_cpp = self._cpp(ecc, omega, us, rs, limb_dark_law)
                    fs_jax = self._jax(ecc, omega, us, rs, limb_dark_law)
                    self.assertTrue(np.all(np.isfinite(fs_jax)))
                    self.assertLess(np.max(np.abs(fs_jax - fs_cpp)), 1.e-9)

    def test_large_planet(self):
        """ Test planets larger than the star against brute-force areas.

        The C++ is not used as a reference here: when the planet covers
        the stellar centre, some of its stellar arcs take the wrong way
        round the star, giving fluxes above one.
        """
        times = np.linspace(4.7, 5.3, 41)
        orbit = [jnp.asarray(x) for x in
                 (self.t0, self.period, self.a, self.inc, 0., 0.)]
        d, _, nu = (np.asarray(x) for x in native._orbit(
            jnp.asarray(times), *orbit, True))
        for n_c in range(3):
            rs = self._random_string(n_c, r0=1.5)
            fs = native.harmonica_transit_quad_ld(
                times, self.t0, self.period, self.a, self.inc, r=rs,
                n_legendre=200)
            brute = np.array([1. - _brute_force_area(d_i, nu_i, rs) / np.pi
                              for d_i, nu_i in zip(d, nu)])
            self.assertLess(np.max(np.abs(np.asarray(fs) - brute)), 1.e-8)

    def test_power2_reduces_to_quadratic(self):
        """ Test power-2 with alpha=2 equals quadratic u1=2c, u2=-c. """
        c = 0.4
        for n_c in range(3):
            rs = self._random_string(n_c)
            fs_p2 = native.harmonica_transit_power2_ld(
                self.times, self.t0, self.period, self.a, self.inc,
                c=c, alpha=2., r=rs)
            fs_quad = native.harmonica_transit_quad_ld(
                self.times, self.t0, self.period, self.a, self.inc,
                u1=2. * c, u2=-c, r=rs)
            self.assertLess(np.max(np.abs(fs_p2 - fs_quad)), 1.e-14)

    def test_gradients(self):
        """ Test autodiff gradients against the C++ analytic derivatives.

        Finite differences are unreliable here: near contact the flux's
        second derivative diverges, so their error does not bound ours.
        """
        from harmonica.jax import custom_primitives
        for limb_dark_law, us in zip(['quadratic', 'non-linear'],
                                     [self.quad_ld, self.nl_ld]):
            if limb_dark_law == 'quadratic':
                fn = native.harmonica_transit_quad_ld
                prim = custom_primitives.jax_light_curve_quad_ld_prim
            else:
                fn = native.harmonica_transit_nonlinear_ld
                prim = custom_primitives.jax_light_curve_nonlinear_ld_prim
            n_ld = len(us)
            for n_c in range(4):
                rs = self._random_string(n_c)
                params = jnp.concatenate([jnp.array(
                    [self.t0, self.period, self.a, self.inc, 0.1, 0.5, *us]),
                    jnp.asarray(rs)])

                @jax.jit
                def flux(p):
                    return fn(self.times, *p[:6 + n_ld], r=p[6 + n_ld:])

                jac = np.asarray(jax.jit(jax.jacfwd(flux))(params))
                self.assertTrue(np.all(np.isfinite(jac)))
                # The C++ custom call only runs on a CPU.
                with jax.default_device(jax.devices('cpu')[0]):
                    _, jac_cpp = prim(self.times, *np.asarray(params))
                # The C++ derivatives of circles with respect to r0 are
                # only accurate to ~2e-6 just inside contact, as checked
                # against converged finite differences.
                self.assertLess(np.max(np.abs(jac - jac_cpp)), 1.e-5)

                # Reverse mode agrees with forward mode.
                grad = np.asarray(jax.grad(
                    lambda p: jnp.sum(flux(p)))(params))
                np.testing.assert_allclose(grad, jac.sum(axis=0),
                                           rtol=1.e-9, atol=1.e-12)

    def test_eccentricity_gradient_at_zero(self):
        """ Test d flux / d ecc at ecc=0 matches a one-sided difference. """
        rs = self._random_string(1)
        omega = 0.3

        def flux(ecc):
            return native.harmonica_transit_quad_ld(
                self.times, self.t0, self.period, self.a, self.inc, ecc,
                omega, *self.quad_ld, r=rs)

        grad = np.asarray(jax.jacfwd(flux)(0.))
        fd = (flux(1.e-7) - flux(0.)) / 1.e-7
        self.assertGreater(np.max(np.abs(grad)), 1.e-3)
        self.assertLess(np.max(np.abs(grad - fd)), 1.e-5)

        # Traced and constant zero eccentricities give the same fluxes.
        np.testing.assert_allclose(jax.jit(flux)(0.), flux(0.), rtol=0.,
                                   atol=1.e-13)

    def test_circle_continuity(self):
        """ Test vanishing asymmetry approaches the circle smoothly. """
        times = np.linspace(4.76, 4.80, 2000)
        circle = native.harmonica_transit_quad_ld(
            times, self.t0, self.period, self.a, self.inc,
            u1=0.1, u2=0.5, r=jnp.array([0.1, 0., 0., 0., 0.]))
        nearly = native.harmonica_transit_quad_ld(
            times, self.t0, self.period, self.a, self.inc,
            u1=0.1, u2=0.5, r=jnp.array([0.1, 1.e-12, 0., 0., -1.e-12]))
        self.assertLess(np.max(np.abs(circle - nearly)), 1.e-11)

    def test_out_of_transit(self):
        """ Test fluxes are exactly one away from transit or behind. """
        times = np.array([0., 2.5, 7.5, 10.])
        fs = native.harmonica_transit_quad_ld(
            times, self.t0, self.period, self.a, self.inc,
            u1=0.1, u2=0.5, r=self._random_string(2))
        np.testing.assert_array_equal(fs, np.ones(4))

    def test_dispatch(self):
        """ Test harmonica.jax's backends.

        The default uses pure jax everywhere; backend='auto' uses the C++
        on CPUs and pure jax elsewhere.
        """
        from harmonica.jax import harmonica_transit_quad_ld
        rs = jnp.asarray(self._random_string(2))

        def default(r):
            return harmonica_transit_quad_ld(
                self.times, self.t0, self.period, self.a, self.inc, 0.1,
                0.2, *self.quad_ld, r=r)

        def flux(r):
            return harmonica_transit_quad_ld(
                self.times, self.t0, self.period, self.a, self.inc, 0.1,
                0.2, *self.quad_ld, r=r, backend='auto')

        fs_native = self._jax(0.1, 0.2, self.quad_ld, rs, 'quadratic')
        np.testing.assert_allclose(jax.jit(default)(rs), fs_native, rtol=0.,
                                   atol=1.e-14)
        if jax.default_backend() == 'cpu':
            ht = HarmonicaTransit(times=self.times)
            ht.set_orbit(self.t0, self.period, self.a, self.inc, 0.1, 0.2)
            ht.set_stellar_limb_darkening(self.quad_ld)
            ht.set_planet_transmission_string(np.asarray(rs))
            np.testing.assert_allclose(jax.jit(flux)(rs),
                                       ht.get_transit_light_curve(),
                                       rtol=0., atol=1.e-15)
        else:
            np.testing.assert_allclose(jax.jit(flux)(rs), fs_native,
                                       rtol=0., atol=1.e-14)

        # Derivatives agree between the two implementations.
        jac = jax.jit(jax.jacfwd(flux))(rs)
        jac_native = jax.jacfwd(lambda r: native.harmonica_transit_quad_ld(
            self.times, self.t0, self.period, self.a, self.inc, 0.1, 0.2,
            *self.quad_ld, r=r))(rs)
        self.assertLess(np.max(np.abs(jac - jac_native)), 1.e-6)

    def test_independent_occulted_areas(self):
        """ Test uniform-star fluxes against brute-force overlap areas.

        Includes planets covering the stellar centre, whose stellar arcs
        cross the far side of the star, and multi-lobed overlaps.
        """
        rng = np.random.default_rng(11)
        n_tested = 0
        while n_tested < 60:
            n_c = int(rng.integers(1, 5))
            r = np.zeros(2 * n_c + 1)
            r[0] = rng.uniform(0.2, 1.2)
            r[1:] = rng.normal(0., 0.35 * r[0] / n_c, 2 * n_c)
            theta = np.linspace(-np.pi, np.pi, 2001)
            rp = np.full_like(theta, r[0])
            for k in range(1, n_c + 1):
                rp += r[2 * k - 1] * np.cos(k * theta) \
                    + r[2 * k] * np.sin(k * theta)
            if rp.min() < 0.02 * r[0]:
                continue
            d = rng.uniform(0., 1. + rp.max())
            nu = rng.uniform(-np.pi, np.pi)
            brute = 1. - _brute_force_area(d, nu, r) / np.pi
            self.assertLess(abs(_uniform_flux(d, nu, r) - brute), 1.e-8)
            n_tested += 1

    def test_tangency_with_crossings(self):
        """ Test a tangency alongside crossings, and a nearby four-root case.

        Both from independent reviews of this implementation.
        """
        r = np.zeros(13)
        r[0], r[11] = 0.125, -0.00390625
        self.assertAlmostEqual(_uniform_flux(1.12109375, 0., r),
                               0.999999975014641, places=13)
        brute = 1. - _brute_force_area(1.12109376, 0., r) / np.pi
        self.assertLess(abs(_uniform_flux(1.12109376, 0., r) - brute), 1.e-9)

    def test_near_contact_first_order_shape(self):
        """ Test a narrow N_c = 1 tip beyond the limb is not missed. """
        r = np.array([0.520001, 0.48 * np.cos(np.pi / 128),
                      0.48 * np.sin(np.pi / 128)])
        thetas, _, count = native._intersections(
            jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(r))
        self.assertEqual(int(count), 2)
        np.testing.assert_allclose(np.asarray(thetas)[:2],
                                   [0.0225024507994, 0.0265849344129],
                                   atol=1.e-12)

    def test_kepler(self):
        """ Test the true anomaly near apastron and as e approaches 1. """
        dsinf = jax.grad(lambda M: native._solve_kepler(M, 0.1)[0])(np.pi)
        self.assertAlmostEqual(float(dsinf),
                               -np.sqrt(1. - 0.01) / (1. + 0.1) ** 2,
                               places=13)
        sinf, cosf, _ = native._solve_kepler(jnp.asarray(1.e-10), 0.99999999)
        np.testing.assert_allclose([sinf, cosf], [0.33507211, -0.94219249],
                                   atol=1.e-8)
        # A tiny negative anomaly keeps its sign.
        sinf, cosf, _ = native._solve_kepler(jnp.asarray(-1.e-18),
                                             1. - 1.e-12)
        np.testing.assert_allclose([sinf, cosf], [-0.899216, 0.437505],
                                   atol=1.e-6)

        # d nu / d omega vanishes at mid-transit.
        def nu(omega):
            return native._orbit(jnp.asarray(5.), *(jnp.asarray(x) for x in (
                5., 10., 10., 1.5, 0.1)), omega, False)[2]
        self.assertLess(abs(float(jax.grad(nu)(-np.pi / 2.))), 1.e-12)

    def test_power2_centre_derivatives(self):
        """ Test second derivatives of the power-2 kernel near the centre.

        With t = 1 - z^2, zeta = sum_j a_j t^(j-1) / k for k = alpha + 2,
        where a_j = (-1)^(j+1) binom(k/2, j).
        """
        k = 3.

        def zeta(t):
            return native._power_zeta(k, jnp.sqrt(1. - t), 1. - t)

        def exact_second_derivative(t):
            a, total = k / 2., 0.
            for j in range(1, 60):
                if j >= 3:
                    total += a * (j - 1) * (j - 2) * t ** (j - 3) / k
                a = -a * (k / 2. - j) / (j + 1.)
            return total
        for t in (0., 1.e-7, 1.e-3, 0.005, 0.02):
            self.assertAlmostEqual(float(jax.grad(jax.grad(zeta))(t)),
                                   exact_second_derivative(t), places=10)

    def test_no_crossing_derivatives(self):
        """ Test derivatives are finite for planets wholly inside the star. """
        def flux(x):
            zero = jnp.asarray(0.)
            s = native._occulted_integrals(
                'quadratic', (zero, zero), x[0], jnp.asarray(1.), x[1],
                x[2:], 50)
            return 1. - s[0] / np.pi
        x = jnp.array([0.1, 0., 0.1])
        np.testing.assert_allclose(jax.grad(flux)(x), [0., 0., -0.2],
                                   atol=1.e-14)
        self.assertAlmostEqual(float(jax.hessian(flux)(x)[2, 2]), -2.,
                               places=12)

    def test_cpu_hessians(self):
        """ Test second derivatives through harmonica.jax on any platform. """
        from harmonica.jax import harmonica_transit_quad_ld

        def flux(a0):
            return harmonica_transit_quad_ld(
                jnp.array([5.]), 5., 10., 10., np.pi / 2., u1=0., u2=0.,
                r=jnp.array([a0]), backend='auto')[0]
        # A uniform star wholly covering the planet: F = 1 - a0^2.
        self.assertAlmostEqual(float(jax.grad(jax.grad(flux))(0.1)), -2.,
                               places=12)

        times = jnp.linspace(4.95, 5.05, 7)

        def dispatched(q):
            return harmonica_transit_quad_ld(
                times, 5., 10., 10., 1.55, u1=q[1], u2=0.2,
                r=jnp.array([q[0], 0.002, -0.001]), backend='auto')

        def pure(q):
            return native.harmonica_transit_quad_ld(
                times, 5., 10., 10., 1.55, u1=q[1], u2=0.2,
                r=jnp.array([q[0], 0.002, -0.001]))
        q = jnp.array([0.1, 0.3])
        np.testing.assert_allclose(
            jax.hessian(lambda x: jnp.sum(dispatched(x)))(q),
            jax.hessian(lambda x: jnp.sum(pure(x)))(q), rtol=0., atol=1.e-6)

        # Time derivatives are supported too.
        np.testing.assert_allclose(
            jnp.diagonal(jax.jacfwd(lambda t: harmonica_transit_quad_ld(
                t, 5., 10., 10., 1.55, u1=0.3, u2=0.2,
                r=jnp.array([0.1, 0.002, -0.001]), backend='auto'))(times)),
            jnp.diagonal(jax.jacfwd(lambda t: native.harmonica_transit_quad_ld(
                t, 5., 10., 10., 1.55, u1=0.3, u2=0.2,
                r=jnp.array([0.1, 0.002, -0.001])))(times)),
            rtol=0., atol=1.e-6)

    def test_dispatch_trailing_zeros(self):
        """ Test strings whose final two coefficients are zero.

        The C++ cannot evaluate them, so the CPU path must not pass them on.
        """
        from harmonica.jax import harmonica_transit_quad_ld
        rs = jnp.array([0.1, 0.002, -0.001, 0., 0.])

        def flux(r):
            return harmonica_transit_quad_ld(
                self.times, self.t0, self.period, self.a, self.inc,
                u1=0.1, u2=0.5, r=r, backend='auto')
        expected = native.harmonica_transit_quad_ld(
            self.times, self.t0, self.period, self.a, self.inc,
            u1=0.1, u2=0.5, r=rs)
        np.testing.assert_allclose(flux(rs), expected, rtol=0., atol=1.e-14)
        np.testing.assert_allclose(
            jax.vmap(flux)(jnp.stack([rs, rs.at[3].set(0.001)]))[0],
            expected, rtol=0., atol=1.e-14)

        # Called directly, the C++ wrapper refuses such strings rather
        # than crashing.
        from harmonica.jax import custom_primitives
        with self.assertRaises(ValueError):
            custom_primitives.harmonica_transit_quad_ld(
                self.times, self.t0, self.period, self.a, self.inc,
                u1=0.1, u2=0.5, r=jnp.array([0.1, 0., 0.]))

    def test_batching(self):
        """ Test per-time parameters, nd times, jit and vmap. """
        rs = jnp.stack([jnp.asarray(self._random_string(1))] * 2)
        times = np.stack([self.times, self.times])
        fs = native.harmonica_transit_quad_ld(
            times, self.t0, self.period, self.a, self.inc,
            u1=jnp.array([[0.1], [0.2]]), u2=0.5, r=rs[:, None, :])
        fs_vmap = jax.jit(jax.vmap(
            lambda u1, r: native.harmonica_transit_quad_ld(
                self.times, self.t0, self.period, self.a, self.inc,
                u1=u1, u2=0.5, r=r)))(jnp.array([0.1, 0.2]), rs)
        self.assertEqual(fs.shape, times.shape)
        np.testing.assert_allclose(fs, fs_vmap, rtol=0., atol=1.e-15)


if __name__ == '__main__':
    unittest.main()
