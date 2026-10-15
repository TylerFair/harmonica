""" Flux accuracy against 500-point quadrature (Table 1) and C++ parity.

For each law and N_c = 0-3, random strings on circular and eccentric
orbits are evaluated by the pure-JAX model at several Gauss-Legendre
orders, and by the C++ at its default (20/50) and at 500 points.
"""
import numpy as np

from common import save
from harmonica import HarmonicaTransit
from harmonica.jax import native

rng = np.random.default_rng(7)
times = np.linspace(-0.2, 0.2, 801)
orders = (20, 50, 100, 200, 500)
results = []
for law, n_u in (('quadratic', 2), ('non-linear', 4)):
    fn = (native.harmonica_transit_quad_ld if law == 'quadratic'
          else native.harmonica_transit_nonlinear_ld)
    for n_c in range(4):
        worst = {f'native_{n}': 0. for n in orders}
        worst['cpp_default'] = 0.
        for trial in range(4):
            r = np.zeros(2 * n_c + 1)
            r[0] = rng.uniform(0.05, 0.2)
            if n_c:
                r[1:] = rng.normal(0., 0.1 * r[0] / n_c, 2 * n_c)
            ecc = 0. if trial % 2 == 0 else 0.2
            orbit = (0., 3.5, 10., np.arccos(rng.uniform(0., 0.9) / 10.),
                     ecc, rng.uniform(0., 2. * np.pi))
            u = list(rng.uniform(0., 0.3, n_u))

            def cpp(pnl_c, pnl_e):
                ht = HarmonicaTransit(times=times, pnl_c=pnl_c, pnl_e=pnl_e)
                ht.set_orbit(*orbit)
                ht.set_stellar_limb_darkening(np.array(u), limb_dark_law=law)
                ht.set_planet_transmission_string(r)
                return ht.get_transit_light_curve()
            reference = cpp(500, 500)
            worst['cpp_default'] = max(worst['cpp_default'], float(
                np.max(np.abs(cpp(20, 50) - reference))))
            for n in orders:
                fs = np.asarray(fn(times, *orbit, *u, r=r, n_legendre=n))
                worst[f'native_{n}'] = max(worst[f'native_{n}'], float(
                    np.max(np.abs(fs - reference))))
        results.append({'law': law, 'n_c': n_c, **worst})
        print(law, n_c, {k: f'{v:.1e}' for k, v in worst.items()}, flush=True)
save('quadrature_accuracy.json', {
    'reference': 'C++ with 500-point quadrature, i.e. HarmonicaTransit('
                 'pnl_c=500, pnl_e=500)', 'cases': results})
