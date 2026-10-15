""" Shared helpers for the reproduction scripts. """
import json
import os
import platform
import subprocess
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))


def brute_force_area(d, nu, r, n=200000):
    """ Overlap of the unit stellar disc with the planet, along rays.

    For each of n angles about the planet's centre, the stellar disc covers
    a radial interval in closed form; integrating rho d(rho) over its overlap
    with [0, r_p(theta)] and summing over angles gives the area.
    """
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    rp = np.full_like(theta, r[0])
    for k in range(1, (len(r) - 1) // 2 + 1):
        rp += r[2 * k - 1] * np.cos(k * theta) + r[2 * k] * np.sin(k * theta)
    c = d * np.cos(theta - nu)
    disc = c * c - d * d + 1.
    root = np.sqrt(np.maximum(disc, 0.))
    lo, hi = np.maximum(c - root, 0.), np.minimum(c + root, rp)
    inside = (disc > 0.) & (hi > lo)
    return np.sum(np.where(inside, 0.5 * (hi * hi - lo * lo), 0.)) \
        * 2. * np.pi / n


def provenance():
    """ Code version and platform, saved with every result. """
    # The code actually imported, which need not be this checkout.
    import harmonica
    repo = os.path.dirname(os.path.dirname(os.path.abspath(harmonica.__file__)))
    commit = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', repo, 'status', '--porcelain',
                            'harmonica'], capture_output=True,
                           text=True).stdout.strip() != ''
    return {'harmonica_commit': commit, 'harmonica_modified': dirty,
            'jax': jax.__version__, 'devices': [str(d) for d in jax.devices()],
            'python': sys.version.split()[0], 'host': platform.node()}


def save(name, payload):
    payload = dict(payload, provenance=provenance())
    path = os.path.join(HERE, 'results', name)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=1, default=float)
    print(f'wrote {path}', flush=True)
