
from typing import Tuple
import numpy as np


def simulate_factor_and_asset_returns(
        t: int=260, k: int=5, n: int=10,
        mu_factor: float=0.05, vol_factor: float=0.20, vol_idio: float=0.10, cash: float=0.03, freq: int=260,
        seed: int=42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate  asset returns based on simulated factor returns
    :param t: number of priods
    :param k: number of factors
    :param n: number of assets
    :param mu_factor: mean excess returns of factors
    :param vol_factor: volatility of factors
    :param vol_idio: volatility of idiosyncratic returns
    :param cash: level of risk returns
    :param freq: frequency per year
    :param seed: default seed
    :return: tuple with factor and asset returns
    """

    # Set seed
    rng = np.random.default_rng(seed)

    # Simulate Factor excess returns
    factors_xret = rng.normal(loc=mu_factor/freq, scale=vol_factor/np.sqrt(freq), size=(t, k))

    # Random Factor exposures
    assets_beta_0 = rng.uniform(low=0.5, high=1.5, size=(n,1))
    assets_beta_1_k = rng.uniform(low=-0.5, high=0.5, size=(n,k-1))
    assets_beta = np.hstack((assets_beta_0, assets_beta_1_k))

    # Idio xret
    idio_xret = rng.normal(loc=0, scale=vol_idio/np.sqrt(freq), size=(t,n))

    # Assets retruns
    assets_xret = factors_xret.dot(assets_beta.T) + idio_xret
    assets_ret = cash/freq + assets_xret

    return assets_ret, assets_xret, factors_xret, assets_beta


