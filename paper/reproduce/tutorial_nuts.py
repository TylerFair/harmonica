""" Re-run harmonica's gradient-based inference tutorial (N_c = 3, NUTS)
with the pure-JAX model, and save the posterior summary and run time.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

from common import save
from harmonica import HarmonicaTransit
from harmonica.jax import harmonica_transit_quad_ld

np.random.seed(12)
times = np.linspace(-0.15, 0.15, 500)
injected_r = np.concatenate([[0.15], np.random.uniform(-0.1, 0.1, 6) * 0.15])
ht = HarmonicaTransit(times)
ht.set_orbit(t0=0., period=4., a=11., inc=87. * np.pi / 180.)
ht.set_stellar_limb_darkening(np.array([0.027, 0.246]), limb_dark_law='quadratic')
ht.set_planet_transmission_string(injected_r)
flux_sigma = 500.e-6 * np.ones(times.shape[0])
observed = ht.get_transit_light_curve() + np.random.normal(0., flux_sigma)


def model(t, f_obs=None):
    r0 = numpyro.sample('r0', dist.Uniform(0.10, 0.20))
    rn_frac = numpyro.sample('rn_frac', dist.Normal(0., 0.1), sample_shape=(6,))
    r = numpyro.deterministic('r', jnp.concatenate([jnp.array([r0]), rn_frac * r0]))
    fs = harmonica_transit_quad_ld(t, t0=0., period=4., a=11.,
                                   inc=87. * np.pi / 180., u1=0.027, u2=0.246, r=r)
    numpyro.sample('obs', dist.Normal(fs, flux_sigma), obs=f_obs)


mcmc = MCMC(NUTS(model, dense_mass=True, max_tree_depth=7,
                 target_accept_prob=0.75, init_strategy=init_to_median()),
            num_warmup=200, num_samples=500, num_chains=2,
            chain_method='sequential', progress_bar=False)
start = time.perf_counter()
mcmc.run(jax.random.PRNGKey(2), times, f_obs=observed)
seconds = time.perf_counter() - start
samples = mcmc.get_samples(group_by_chain=True)['r']
summary = numpyro.diagnostics.summary(samples)
save(f'tutorial_nuts_{jax.devices()[0].platform}.json', {
    'seconds_including_compile': seconds, 'injected_r': injected_r.tolist(),
    'mean': summary['Param:0']['mean'].tolist(),
    'std': summary['Param:0']['std'].tolist(),
    'r_hat': summary['Param:0']['r_hat'].tolist(),
    'n_eff': summary['Param:0']['n_eff'].tolist(),
    'published_notebook_mean': [0.15018, -0.00783, 0.00691, -0.00808,
                                0.00203, -0.01326, 0.01209],
    'published_notebook_sd': [0.00040, 0.00126, 0.00159, 0.00554, 0.00192,
                              0.00508, 0.00363]})
