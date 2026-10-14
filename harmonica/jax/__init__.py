__all__ = [
    'harmonica_transit_quad_ld',
    'harmonica_transit_nonlinear_ld',
    'native',
]

from . import native


def __getattr__(name):
    # The custom-call primitives are imported on first use, so that the
    # pure-jax module does not depend on their jax internals.
    if name in ('harmonica_transit_quad_ld', 'harmonica_transit_nonlinear_ld'):
        from . import custom_primitives
        return getattr(custom_primitives, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
