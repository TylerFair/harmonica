""" Uniform-star fluxes against brute-force overlap areas (Section 4).

Draws random positive-radius strings, N_c = 1-4 and a_0 = 0.2-1.2, at
separations anywhere up to contact, including planets that cover the
stellar centre. Flagged differences are recomputed with 40 times more
rays, to separate reference error from model error.
"""
import numpy as np
import jax
import jax.numpy as jnp

from common import brute_force_area, save
from harmonica.jax import native

n_shapes = 6000
zero = jnp.asarray(0.)
occulted = jax.jit(lambda d, nu, r: native._occulted_integrals(
    'quadratic', (zero, zero), d, jnp.asarray(1.), nu, r, 200)[0])
rng = np.random.default_rng(1)
rows, tested = [], 0
while tested < n_shapes:
    n_c = int(rng.integers(1, 5))
    r = np.zeros(2 * n_c + 1)
    r[0] = rng.uniform(0.2, 1.2)
    r[1:] = rng.normal(0., 0.35 * r[0] / n_c, 2 * n_c)
    theta = np.linspace(-np.pi, np.pi, 4001)
    rp = np.full_like(theta, r[0])
    for k in range(1, n_c + 1):
        rp += r[2 * k - 1] * np.cos(k * theta) + r[2 * k] * np.sin(k * theta)
    if rp.min() < 0.02 * r[0]:
        continue
    d, nu = rng.uniform(0., 1. + rp.max()), rng.uniform(-np.pi, np.pi)
    area = float(occulted(jnp.asarray(d), jnp.asarray(nu), jnp.asarray(r)))
    rows.append((d, nu, r, area, brute_force_area(d, nu, r)))
    tested += 1
errors = np.array([abs(a - b) / np.pi for _, _, _, a, b in rows])
flagged = np.flatnonzero(errors > 1.e-8)
recheck = []
for i in flagged:
    d, nu, r, area, _ = rows[i]
    fine = brute_force_area(d, nu, r, n=8000000)
    recheck.append({'d': d, 'nu': nu, 'r': r.tolist(),
                    'flux_error_2e5_rays': errors[i],
                    'flux_error_8e6_rays': abs(area - fine) / np.pi})
print(f'{n_shapes} shapes: max flux error {errors.max():.2e} at 2e5 rays; '
      f'{len(flagged)} above 1e-8, max {max([c["flux_error_8e6_rays"] for c in recheck], default=0.):.2e} at 8e6 rays')
save('brute_force_areas.json', {
    'n_shapes': n_shapes, 'max_flux_error_2e5_rays': errors.max(),
    'n_flagged_above_1e-8': len(flagged), 'flagged_rechecked': recheck,
    'max_flux_error_flagged_8e6_rays': max(
        [c['flux_error_8e6_rays'] for c in recheck], default=0.)})
