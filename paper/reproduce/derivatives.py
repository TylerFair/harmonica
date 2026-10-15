""" Derivative checks (Section 4).

Forward against reverse mode; the pure-JAX model against converged finite
differences just inside contact, where the C++ derivative of a circle with
respect to r0 is off; the C++ path's second and time derivatives; and the
true anomaly near apastron and at extreme eccentricity.
"""
import numpy as np
import jax
import jax.numpy as jnp

from common import save
from harmonica.jax import harmonica_transit_quad_ld, native, custom_primitives

out = {}
times = jnp.linspace(4.7, 5.3, 1000)
inc = 89. * np.pi / 180.


def native_flux(p):
    return native.harmonica_transit_quad_ld(times, *p[:8], r=p[8:])


p = jnp.array([5., 10., 10., inc, 0.1, 0.5, 0.1, 0.5, 0.1, 0.002, -0.001,
               0.001, 0.0005])
jac = jax.jacfwd(native_flux)(p)
grad = jax.grad(lambda q: jnp.sum(native_flux(q)))(p)
out['forward_vs_reverse'] = float(jnp.max(jnp.abs(grad - jac.sum(0))))

# Circle just inside second contact: AD, the C++ and finite differences.
contact = []
for i in (724, 725, 726):
    t = jnp.array([np.linspace(4.7, 5.3, 1000)[i]])

    def circle(r0):
        return native.harmonica_transit_quad_ld(
            t, 5., 10., 10., inc, 0.1, 0.5, 0.1, 0.5, r=jnp.array([r0]))[0]
    prim = custom_primitives.jax_light_curve_quad_ld_prim(
        t, 5., 10., 10., inc, 0.1, 0.5, 0.1, 0.5, 0.1)
    fd = [float((circle(0.1 + h) - circle(0.1 - h)) / (2 * h))
          for h in (1e-5, 1e-6, 1e-7)]
    contact.append({'index': i, 'native_ad': float(jax.grad(circle)(0.1)),
                    'cpp_analytic': float(prim[1][0, 8]),
                    'finite_differences_h_1e-5_1e-6_1e-7': fd})
out['circle_r0_near_contact'] = contact

# The C++ path's Hessians and time derivatives, against the native model.


def cpp_flux(a0):
    return harmonica_transit_quad_ld(
        jnp.array([5.]), 5., 10., 10., np.pi / 2., u1=0., u2=0.,
        r=jnp.array([a0]), backend='auto')[0]
out['cpp_path_internal_circle_d2F'] = float(jax.grad(jax.grad(cpp_flux))(0.1))
short = jnp.linspace(4.95, 5.05, 7)


def model(q, backend):
    return harmonica_transit_quad_ld(
        short, 5., 10., 10., 1.55, u1=q[1], u2=0.2,
        r=jnp.array([q[0], 0.002, -0.001]), backend=backend)
q = jnp.array([0.1, 0.3])
out['cpp_path_hessian_vs_native'] = float(jnp.max(jnp.abs(
    jax.hessian(lambda x: jnp.sum(model(x, 'auto')))(q)
    - jax.hessian(lambda x: jnp.sum(model(x, 'native')))(q))))

# Eccentricity derivative at e = 0, against finite differences.


def ecc_flux(e):
    return harmonica_transit_quad_ld(
        times, 5., 10., 10., 1.55, e, 0.3, 0.1, 0.5,
        r=jnp.array([0.1, 0.002, -0.001]))
out['default_decc_at_0_vs_fd'] = float(jnp.max(jnp.abs(
    jax.jacfwd(ecc_flux)(0.) - (ecc_flux(1e-7) - ecc_flux(0.)) / 1e-7)))

# Orbits.
out['dsinf_dM_apastron_e0.1'] = float(jax.grad(
    lambda M: native._solve_kepler(M, 0.1)[0])(np.pi))
out['dsinf_dM_apastron_e0.1_exact'] = -np.sqrt(1. - 0.01) / 1.1 ** 2
sinf, cosf, _ = native._solve_kepler(jnp.asarray(1.e-10), 0.99999999)
out['true_anomaly_e0.99999999_M1e-10'] = [float(sinf), float(cosf)]
print(out, flush=True)
save('derivatives.json', out)
