""" Timings (Section 5, Table 2): pure JAX on the default device against
the C++ on one core of the same node.

Run on an otherwise idle GPU for meaningful numbers.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from common import save
from harmonica import HarmonicaTransit
from harmonica.jax import native


def timed(fn, *args, repeats=20):
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - start) / repeats * 1e3


orbit = (0., 3.5, 10., np.arccos(0.3 / 10.), 0., 0.)
table = []
for n_times in (1000, 10000):
    times = np.linspace(-0.15, 0.15, n_times)
    for n_c in (1, 2, 3):
        r = np.zeros(2 * n_c + 1)
        r[0], r[1:] = 0.12, 0.004 * (-1.) ** np.arange(2 * n_c)
        ht = HarmonicaTransit(times=times)
        ht.set_orbit(*orbit)
        ht.set_stellar_limb_darkening(np.array([0.3, 0.2]))
        ht.set_planet_transmission_string(r)
        start = time.perf_counter()
        for _ in range(5):
            ht.get_transit_light_curve()
        cpp_ms = (time.perf_counter() - start) / 5 * 1e3
        value = jax.jit(lambda rr: native.harmonica_transit_quad_ld(
            times, *orbit, 0.3, 0.2, r=rr))
        grad = jax.jit(jax.grad(lambda rr: jnp.sum(
            native.harmonica_transit_quad_ld(times, *orbit, 0.3, 0.2, r=rr))))
        table.append({'n_times': n_times, 'n_c': n_c, 'cpp_1core_ms': cpp_ms,
                      'native_ms': timed(value, jnp.asarray(r)),
                      'native_value_grad_ms': timed(grad, jnp.asarray(r))})
        print(table[-1], flush=True)

# Order-six, cosine-only strings, as in the WASP-17b analysis.
P, a, inc = 3.73548556, 7.148, np.radians(87.361)
times = (np.arange(582) - 291) * 60. / 86400.
rng = np.random.default_rng(0)


def strings(n):
    r = np.zeros((n, 13))
    r[:, 0] = 0.1234
    r[:, 1:12:2] = 0.0015 * rng.standard_normal((n, 6)) / np.arange(1, 7)
    return r
ht = HarmonicaTransit(times=times)
ht.set_orbit(0., P, a, inc)
ht.set_stellar_limb_darkening(np.array([0.1, 0.25]))
r_cpp = strings(100)
start = time.perf_counter()
for k in range(100):
    ht.set_planet_transmission_string(r_cpp[k])
    ht.get_transit_light_curve()
order_six = {'cpp_1core_ms_per_curve': (time.perf_counter() - start) / 100 * 1e3}
batch = jax.jit(lambda rr: native.harmonica_transit_quad_ld(
    np.broadcast_to(times, (rr.shape[0], times.size)), 0., P, a, inc,
    u1=0.1, u2=0.25, r=rr[:, None, :]))
for n in (1, 26, 1560, 3120):
    ms = timed(batch, jnp.asarray(strings(n)), repeats=5 if n > 100 else 20)
    order_six[f'native_batch_{n}_ms_per_curve'] = ms / n
print(order_six, flush=True)
save(f'gpu_benchmark_{jax.devices()[0].device_kind.replace(" ", "_")}.json',
     {'table_2': table, 'order_six_wasp17_like': order_six})
