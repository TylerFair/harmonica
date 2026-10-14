import unittest
import numpy as np
import jax
import jax.numpy as jnp

from harmonica import HarmonicaTransit
from harmonica.jax import native


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

    def test_large_planet_parity(self):
        """ Test fluxes match the C++ when the planet covers the star. """
        for n_c in range(3):
            rs = self._random_string(n_c, r0=1.5)
            fs_cpp = self._cpp(0., 0., self.quad_ld, rs, 'quadratic')
            fs_jax = self._jax(0., 0., self.quad_ld, rs, 'quadratic')
            self.assertLess(np.max(np.abs(fs_jax - fs_cpp)), 1.e-9)

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
        """ Test autodiff gradients against finite differences. """
        for n_c in range(3):
            rs = self._random_string(n_c)
            params = jnp.concatenate([jnp.array(
                [self.t0, self.period, self.a, self.inc, 0.1, 0.5,
                 *self.quad_ld]), jnp.asarray(rs)])

            @jax.jit
            def flux(p):
                return native.harmonica_transit_quad_ld(
                    self.times, *p[:8], r=p[8:])

            jac = np.asarray(jax.jit(jax.jacfwd(flux))(params))
            self.assertTrue(np.all(np.isfinite(jac)))

            # Reverse mode agrees with forward mode.
            grad = np.asarray(jax.grad(lambda p: jnp.sum(flux(p)))(params))
            np.testing.assert_allclose(grad, jac.sum(axis=0), rtol=1.e-9,
                                       atol=1.e-12)

            for i in range(params.shape[0]):
                h = 1.e-6 * max(1., abs(float(params[i])))
                fd = (flux(params.at[i].add(h))
                      - flux(params.at[i].add(-h))) / (2. * h)
                self.assertLess(np.max(np.abs(jac[:, i] - fd)), 1.e-5)

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
