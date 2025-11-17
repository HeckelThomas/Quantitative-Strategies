
# ------------------------------
# 4) Ledoit–Wolf shrinkage covariance
# Problem:
#  - Implement Ledoit & Wolf analytic shrinkage estimator.
#
# Math summary:
#   - Sample covariance S = 1/n sum (x_t - mu)(x_t - mu)^T
#   - Target F = muI where mu = trace(S)/p
#   - Shrunk = delta F + (1-delta) S
#   - Delta computed using formula: delta_hat = (pi_hat / rho_hat) / n (clipped in [0,1])
#   - pi_hat = (1/n) sum_t ||x_t x_t^T - S||_F^2
#   - rho_hat = ||S - F||_F^2
#
# Implementation below.
# ------------------------------

import numpy as np
from typing import Tuple
from src.vcv.utils import *

def vcv_estimation(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Ledoit-Wolf shrinkage estimator based on Returns
    :param returns: Asset Returns
    :return: LW vcv, vol, correl + Empirical vcv, vol, correl + delta
    """
    # Empirical vcv and correl matrix
    vcv_emp = np.cov(returns.T)
    vol_emp = np.sqrt(np.diag(vcv_emp))
    correl_emp = np.diag(1/vol_emp).dot(vcv_emp).dot(np.diag(1/vol_emp))
    # f matrix with average variance for all assets
    t = returns.shape[0]
    n = returns.shape[1]
    f = np.identity(n)*np.diag(vcv_emp).sum()/n
    # Weight of empirical vcv and f matrix
    centered = returns-returns.mean(axis=0, keepdims=True)
    pi_hat = 0
    for i in range(t):
        x_i = centered[i,:].reshape(-1,1)
        pi_hat += ((x_i.dot(x_i.T)-vcv_emp)**2).sum()
    pi_hat /= t
    rho_hat = ((f-vcv_emp)**2).sum()
    delta = pi_hat/rho_hat/t
    delta = max(min(delta, 1.0), 0.0)
    # Ledoit and Wolf shrinkage vcv
    vcv_lw = delta*f + (1-delta)*vcv_emp
    vol_lw = np.sqrt(np.diag(vcv_lw))
    correl_lw = np.diag(1/vol_lw).dot(vcv_lw).dot(np.diag(1/vol_lw))
    return vcv_lw, vol_lw, correl_lw, vcv_emp, vol_emp, correl_emp, delta




