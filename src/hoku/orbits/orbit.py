import jax
import jax.numpy as jnp
import equinox as eqx
from exoplanet_core.jax import ops


class Orbit(eqx.Module):
    period: jax.Array
    t_periastron: jax.Array
    omega: jax.Array
    ecc: jax.Array

    t0: jax.Array
    cos_omega: jax.Array
    sin_omega: jax.Array
    M0: jax.Array
    n: jax.Array

    def __init__(self, period=None, t_periastron=None, omega=None, ecc=None):
        self.period = period
        self.t_periastron = t_periastron
        self.omega = omega
        self.ecc = ecc

        self.sin_omega = jnp.sin(self.omega)
        self.cos_omega = jnp.cos(self.omega)

        # Some things
        opsw = 1 + self.sin_omega
        E0 = 2 * jnp.arctan2(
            jnp.sqrt(1 - self.ecc) * self.cos_omega,
            jnp.sqrt(1 + self.ecc) * opsw,
        )
        self.M0 = E0 - self.ecc * jnp.sin(E0)
        self.n = 2 * jnp.pi / self.period
        self.t0 = self.t_periastron + self.M0 / self.n

    def _get_true_anomaly(self, t):
        M = 2.0 * jnp.pi * (t - self.t_periastron) / self.period
        sinf, cosf = ops.kepler(M, self.ecc + jnp.zeros_like(M))
        return jnp.arctan2(sinf, cosf)

    def get_radial_velocity(self, t, K):
        f = self._get_true_anomaly(t)
        cosf = jnp.cos(f)
        sinf = jnp.sin(f)

        return jnp.squeeze(
            K
            * (
                jnp.cos(self.omega) * cosf
                - jnp.sin(self.omega) * sinf
                + self.ecc * jnp.cos(self.omega)
            )
        )
