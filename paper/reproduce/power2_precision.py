""" The power-2 radial kernel against a 75-digit reference (Section 3.4).

zeta(t) = [1 - (1 - t)^p] / (2 p t), with p = (alpha + 2) / 2, is summed
from its Taylor series in exact decimal arithmetic, and compared with the
model's value and its first and second derivatives.
"""
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp

from common import save
from harmonica.jax import native


def reference(t, k, m):
    """ The m-th derivative of zeta at t, to 75 digits. """
    with localcontext() as ctx:
        ctx.prec = 75
        t, k = Decimal(repr(t)), Decimal(repr(k))
        coeff, total = Decimal('0.5'), Decimal(0)
        for i in range(400):
            if i >= m:
                factor = 1
                for j in range(m):
                    factor *= i - j
                total += coeff * factor * t ** (i - m)
            coeff = coeff * (Decimal(i + 1) - k / 2) / Decimal(i + 2)
        return float(total)


def zeta(t, k):
    return native._power_zeta(k, jnp.sqrt(1. - t), 1. - t)


derivatives = [jax.jit(zeta), jax.jit(jax.grad(zeta)),
               jax.jit(jax.grad(jax.grad(zeta)))]
rows = []
for alpha in (0.01, 0.5, 1., 2., 5., 10., 20., 100.):
    k = alpha + 2.
    for t in (0., 1.e-9, 1.e-6, 1.e-4, 1.e-3, 0.00999, 0.01, 0.0101, 0.05,
              0.1, 0.3, 0.49):
        errors = [abs(float(f(t, k)) - reference(t, k, m))
                  for m, f in enumerate(derivatives)]
        rows.append({'alpha': alpha, 't': t, 'value_error': errors[0],
                     'first_derivative_error': errors[1],
                     'second_derivative_error': errors[2]})
worst = {key: max(r[key] for r in rows) for key in
         ('value_error', 'first_derivative_error', 'second_derivative_error')}
print(worst, flush=True)
save('power2_precision.json', {'worst': worst, 'cases': rows})
