
# quant_interview_set3_full.py
"""
Quant Interview — Set 3 (Problems + Detailed Math + Solutions)
Author: ChatGPT
Created: 2025-11-08

This file contains 10 interview-style exercises for senior quantitative roles.
Each exercise includes:
 - Problem statement
 - Mathematical explanation / derivation
 - Python implementation (clear, idiomatic, documented)
 - A small test or usage example

Requirements:
 - numpy, scipy
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from scipy import stats

# ------------------------------
# Utilities
# ------------------------------
def annualize_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (mean / std) using simple annualization."""
    mean = returns.mean() * periods_per_year
    sd = returns.std(ddof=1) * math.sqrt(periods_per_year)
    return float(mean / sd) if sd > 0 else 0.0

# ------------------------------
# 1) Autocovariance & AR(1) estimation (Yule-Walker and conditional MLE)
# Problem:
#   - Compute empirical autocovariances up to lag L.
#   - Estimate AR(1) phi using Yule-Walker phi = gamma(1)/gamma(0).
#   - Estimate phi by conditional MLE: regress x_t on x_{t-1} (no intercept).
#
# Math:
#   x_t = phi * x_{t-1} + eps_t, eps_t ~ N(0, sigma^2).
#   Yule-Walker: gamma(k) = E[x_t x_{t-k}] ; phi_hat = gamma_hat(1)/gamma_hat(0).
#   Conditional MLE (OLS): phi_hat = sum_{t=1..n-1} x_t x_{t-1} / sum_{t=1..n-1} x_{t-1}^2.
#
# Implementation below.
# ------------------------------
def autocovariances(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    mu = x.mean()
    out = np.empty(L+1, dtype=float)
    for k in range(L+1):
        denom = n - k
        out[k] = ((x[:denom] - mu) * (x[k:k+denom] - mu)).sum() / denom
    return out

def ar1_yule_walker(x: np.ndarray) -> Tuple[float, float]:
    g = autocovariances(x, 1)
    phi = g[1] / g[0] if g[0] != 0 else 0.0
    sigma2 = g[0] * (1 - phi*phi)
    return float(phi), float(sigma2)

def ar1_conditional_mle(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return 0.0, 0.0
    num = (x[1:] * x[:-1]).sum()
    den = (x[:-1]**2).sum()
    phi = num / den if den != 0 else 0.0
    resid = x[1:] - phi * x[:-1]
    sigma2 = (resid**2).sum() / (n - 1)
    return float(phi), float(sigma2)

def _example_ar1():
    rng = np.random.default_rng(0)
    phi_true = 0.7
    sigma_true = 0.5
    n = 1000
    x = np.empty(n)
    x[0] = rng.normal(scale=sigma_true / math.sqrt(1-phi_true**2))
    for t in range(1, n):
        x[t] = phi_true * x[t-1] + rng.normal(scale=sigma_true)
    print("Yule-Walker:", ar1_yule_walker(x)[0])
    print("Conditional MLE:", ar1_conditional_mle(x)[0])

# ------------------------------
# 2) Pair trading: cointegration (Engle–Granger) and simple z-score strategy
# Problem:
#  - Given two price series p1, p2, test cointegration via Engle-Granger:
#      regress p1_t = a + b p2_t + u_t, test u_t is stationary (ADF on residuals).
#  - Implement simple z-score pair strategy on residuals.
#
# Math:
#  - Regression yields residuals u_t = p1_t - a - b p2_t.
#  - Compute rolling mean m_t and std s_t of u_t; z_t = (u_t - m_t)/s_t.
#  - Trading rule: if z_t > entry -> short spread, if z_t < -entry -> long spread; exit at z near 0.
#
# Implementation: regression (numpy lstsq), ADF test via statsmodels would be ideal,
# but here we use a simple Augmented Dickey-Fuller proxy: regress Δu_t on u_{t-1}.
# ------------------------------
def engle_granger_adf_proxy(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    # regression p1 ~ 1 + p2
    X = np.vstack([np.ones(p2.size), p2]).T
    beta = np.linalg.lstsq(X, p1, rcond=None)[0]
    resid = p1 - X.dot(beta)
    # compute DF statistic for resid: regress delta_resid on resid_lag (no intercept)
    d = resid[1:] - resid[:-1]
    r = resid[:-1]
    den = (r**2).sum()
    num = (r * d).sum()
    phi = num / den if den != 0 else 0.0
    df_stat = (phi - 1) * math.sqrt(len(r))
    pval = 2 * (1 - stats.norm.cdf(abs(df_stat)))
    return float(df_stat), float(pval)

def rolling_zscore(resid: np.ndarray, window: int = 252) -> np.ndarray:
    resid = np.asarray(resid, dtype=float)
    n = resid.size
    if window < 2:
        raise ValueError("window >= 2")
    means = np.convolve(resid, np.ones(window)/window, mode='valid')
    means_full = np.concatenate([np.full(window-1, np.nan), means])
    # compute rolling std properly
    s2 = np.convolve(resid**2, np.ones(window)/window, mode='valid') - means**2
    s2 = np.maximum(s2, 1e-12)
    s_full = np.concatenate([np.full(window-1, np.nan), np.sqrt(s2)])
    z = (resid - means_full) / s_full
    return z

def pair_trading_backtest(p1: np.ndarray, p2: np.ndarray, entry: float = 1.0, exit: float = 0.0, tc: float = 0.0005) -> Dict[str, Any]:
    X = np.vstack([np.ones(p2.size), p2]).T
    beta = np.linalg.lstsq(X, p1, rcond=None)[0]
    resid = p1 - X.dot(beta)
    z = rolling_zscore(resid, window=min(252, len(resid)//2))
    pos = np.zeros_like(z)
    in_pos = 0  # 0 flat, +1 long spread (buy p1, sell p2), -1 short spread
    for i in range(len(z)):
        if np.isnan(z[i]):
            pos[i] = 0
            continue
        if in_pos == 0:
            if z[i] > entry:
                in_pos = -1
            elif z[i] < -entry:
                in_pos = 1
        else:
            if abs(z[i]) <= exit:
                in_pos = 0
        pos[i] = in_pos
    # approximate daily PnL: pos * change in spread normalized
    spread = resid
    spread_ret = np.concatenate(([0.0], np.diff(spread)))  # absolute changes in spread
    pnl = pos * spread_ret
    turnover = np.concatenate(([0.0], np.abs(np.diff(pos))))
    pnl_net = pnl - turnover * tc
    sharpe = annualize_sharpe(np.array(pnl_net))
    return {"pnl": pnl_net, "sharpe": sharpe, "beta": beta}

def _example_pair():
    rng = np.random.default_rng(1)
    T = 1000
    p2 = np.cumsum(rng.normal(scale=0.5, size=T)) + 50
    # make p1 cointegrated: p1 = 2 * p2 + small noise
    p1 = 2.0 * p2 + np.cumsum(rng.normal(scale=0.1, size=T))
    res = pair_trading_backtest(p1, p2)
    print("Pair Sharpe (approx):", res["sharpe"])

# ------------------------------
# 3) Online PCA: incremental rank-1 updates (pedagogical)
# Problem:
#  - Maintain top-k principal components when new samples arrive.
#
# Math idea:
#  - Keep orthonormal matrix U (k x d). For a new sample x, project p = U x.
#  - Update U toward the direction of x's contribution. Oja's rule or incremental SVD can be used.
#
# Implementation below uses a simple normalized gradient ascent-ish update and Gram-Schmidt to keep U orthonormal.
# ------------------------------
class OnlinePCA:
    def __init__(self, k: int, dim: int, lr: float = 0.1):
        self.k = k
        self.dim = dim
        self.lr = float(lr)
        self.U = np.zeros((k, dim))
        for i in range(min(k, dim)):
            e = np.zeros(dim); e[i] = 1.0
            self.U[i] = e
        self.n = 0

    def partial_fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError("dimension mismatch")
        self.n += 1
        p = self.U.dot(x)            # projections (k,)
        recon = p.dot(self.U)        # reconstruction
        residual = x - recon
        eta = self.lr / max(1.0, math.sqrt(self.n))
        for i in range(self.k):
            u = self.U[i]
            delta = eta * (p[i] * x - (p[i]**2) * u)
            u = u + delta
            norm = np.linalg.norm(u)
            if norm > 0:
                u /= norm
            self.U[i] = u
        # Gram-Schmidt orthonormalize
        for i in range(self.k):
            for j in range(i):
                self.U[i] -= self.U[j].dot(self.U[i]) * self.U[j]
            nrm = np.linalg.norm(self.U[i])
            if nrm > 0:
                self.U[i] /= nrm

    @property
    def components_(self):
        return self.U.copy()

def _example_online_pca():
    rng = np.random.default_rng(0)
    d = 10; k = 3
    pca = OnlinePCA(k, d)
    X = rng.normal(size=(500, d))
    for x in X:
        pca.partial_fit(x)
    print("Components shape:", pca.components_.shape)

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
def ledoit_wolf(X: np.ndarray) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T.dot(Xc)) / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    # compute pi_hat
    pi_hat = 0.0
    for i in range(n):
        xi = Xc[i][:, None]
        outer = xi.dot(xi.T)
        pi_hat += ((outer - S) ** 2).sum()
    pi_hat /= n
    print(pi_hat)
    rho_hat = ((S - F) ** 2).sum()
    print(rho_hat)
    delta = min(max(pi_hat / (rho_hat * n) if rho_hat > 0 else 0.0, 0.0), 1.0)
    Sigma_hat = delta * F + (1 - delta) * S
    return Sigma_hat, float(delta)

def _example_lw():
    rng = np.random.default_rng(1)
    p = 20; n = 200
    A = rng.normal(size=(p, p))
    true_cov = A.dot(A.T)
    X = rng.multivariate_normal(np.zeros(p), true_cov, size=n)
    S_shr, delta = ledoit_wolf(X)
    print("Delta:", delta)

# ------------------------------
# 5) Factor regression + cross-sectional shrinkage
# Problem:
#  - Given returns R (T x N), factors F (T x K), estimate betas for each asset via OLS and shrink across cross-section toward mean.
#
# Math:
#  - For each asset i: regress R_i = alpha_i + B_i * F + eps_i.
#  - Let B = [B_1; ...; B_N], shrink: B_shr = (1 - λ) B + λ * mean(B) (applied row-wise toward cross-sectional mean).
#
# Implementation: class FactorRegressor with fit and predict.
# ------------------------------
@dataclass
class FactorRegressor:
    lambd: float = 0.1
    betas_: Optional[np.ndarray] = None  # N x K
    intercepts_: Optional[np.ndarray] = None
    beta_mean_: Optional[np.ndarray] = None

    def fit(self, R: np.ndarray, F: np.ndarray) -> "FactorRegressor":
        # R: T x N, F: T x K
        T, N = R.shape
        K = F.shape[1]
        X = np.hstack([np.ones((T,1)), F])  # T x (K+1)
        betas = np.zeros((N, K))
        intercepts = np.zeros(N)
        for i in range(N):
            y = R[:, i]
            sol = np.linalg.lstsq(X, y, rcond=None)[0]
            intercepts[i] = sol[0]
            betas[i] = sol[1:]
        beta_mean = betas.mean(axis=0)
        betas_shr = (1 - self.lambd) * betas + self.lambd * beta_mean[None, :]
        self.betas_ = betas_shr
        self.intercepts_ = intercepts
        self.beta_mean_ = beta_mean
        return self

    def predict(self, F_new: np.ndarray) -> np.ndarray:
        if self.betas_ is None or self.intercepts_ is None:
            raise ValueError("call fit first")
        T = F_new.shape[0]
        Xn = np.hstack([np.ones((T,1)), F_new])  # T x (K+1)
        coefs = np.hstack([self.intercepts_[:, None], self.betas_])  # N x (K+1)
        return Xn.dot(coefs.T)  # T x N

def _example_factor():
    rng = np.random.default_rng(2)
    T, N, K = 200, 50, 3
    F = rng.normal(size=(T, K))
    true_betas = rng.normal(size=(N, K))
    intercepts = rng.normal(size=N)
    R = F.dot(true_betas.T) + intercepts + 0.1 * rng.normal(size=(T, N))
    fr = FactorRegressor(lambd=0.2).fit(R, F)
    preds = fr.predict(F)
    print("Preds shape:", preds.shape)

# ------------------------------
# 6) P² algorithm: online quantile estimation (Percentile algorithm)
# Problem:
#  - Implement P² algorithm to estimate a running quantile (e.g., median) in O(1) memory/time per update.
#
# Math (high level):
#  - Maintain five markers (positions and heights) corresponding to min, quantile estimates, median, max etc.
#  - Update marker heights using parabolic formula when the desired position shifts enough; else linear.
#
# Implementation below is a compact implementation for single quantile p (0<p<1).
# ------------------------------
class P2Quantile:
    def __init__(self, p: float = 0.5):
        if not (0 < p < 1):
            raise ValueError("p in (0,1)")
        self.p = p
        self.n = 0
        self.initial: List[float] = []
        self.q = [0.0] * 5
        self.ni = [1, 2, 3, 4, 5]  # positions of markers
        self.np = [0, p/2, p, (1+p)/2, 1]  # desired positions normalized

    def add(self, x: float):
        x = float(x)
        self.n += 1
        if self.n <= 5:
            self.initial.append(x)
            if self.n == 5:
                self.initial.sort()
                self.q = self.initial.copy()
            return
        # find k
        k = 0
        while k < 4 and x >= self.q[k+1]:
            k += 1
        if x < self.q[0]:
            self.q[0] = x
            k = 0
        if x > self.q[4]:
            self.q[4] = x
            k = 3
        for i in range(5):
            if i > k:
                self.ni[i] += 1
        # desired positions in counts
        ns = [1, 1 + 2*self.p, 1 + 4*self.p, 3 + 2*self.p, 5]  # simple approx for initial; not strictly increasing with n
        # adjust internal markers
        for i in range(1,4):
            d = ns[i] - self.ni[i]
            if (d >= 1 and (self.q[i+1] - self.q[i]) > 0) or (d <= -1 and (self.q[i] - self.q[i-1]) > 0):
                d_sign = int(math.copysign(1, d))
                q_ip1 = self.q[i+1]; q_i = self.q[i]; q_im1 = self.q[i-1]
                n_ip1 = self.ni[i+1]; n_i = self.ni[i]; n_im1 = self.ni[i-1]
                den = n_ip1 - n_im1
                if den == 0:
                    continue
                num = d_sign * (n_i - n_im1 + d_sign) * (q_ip1 - q_i) / (n_ip1 - n_i)
                num += d_sign * (n_ip1 - n_i - d_sign) * (q_i - q_im1) / (n_i - n_im1)
                q_new = q_i + num / den
                if q_im1 < q_new < q_ip1:
                    self.q[i] = q_new
                else:
                    self.q[i] += d_sign * (q_ip1 - q_im1) / (n_ip1 - n_im1)
                self.ni[i] += d_sign

    def quantile(self) -> float:
        if self.n <= 5:
            return float(np.median(self.initial))
        return float(self.q[2])

def _example_p2():
    rng = np.random.default_rng(3)
    data = rng.normal(size=1000)
    p2 = P2Quantile(0.5)
    for x in data:
        p2.add(x)
    print("P2 median approx:", p2.quantile(), "true median:", np.median(data))

# ------------------------------
# 7) Hidden Markov Model (Gaussian emissions): Viterbi and Baum–Welch (EM)
# Problem:
#  - Implement HMM with Gaussian emissions; methods: viterbi(obs) and fit_bw(obs).
#
# Math:
#  - Forward/backward in log-space for numerical stability.
#  - E-step: compute gamma (state posteriors) and xi (transition posteriors).
#  - M-step: update pi, A, means, variances using weighted sums.
#
# Implementation below provides a reasonably clear, vectorized version.
# ------------------------------
class GaussianHMM:
    def __init__(self, n_states: int, var_shared: Optional[float] = None):
        self.n = n_states
        self.pi = np.full(self.n, 1.0 / self.n)
        self.A = np.full((self.n, self.n), 1.0 / self.n)
        self.means = np.linspace(-1.0, 1.0, self.n)
        self.vars = np.full(self.n, 1.0) if var_shared is None else np.full(self.n, var_shared)
        self.var_shared = var_shared

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        T = len(obs)
        out = np.empty((T, self.n))
        for i in range(self.n):
            out[:, i] = stats.norm.logpdf(obs, loc=self.means[i], scale=math.sqrt(self.vars[i]))
        return out

    def viterbi(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        T = len(obs)
        logB = self._log_emission(obs)
        logA = np.log(self.A + 1e-12)
        logpi = np.log(self.pi + 1e-12)
        dp = np.full((T, self.n), -1e30)
        ptr = np.zeros((T, self.n), dtype=int)
        dp[0] = logpi + logB[0]
        for t in range(1, T):
            for j in range(self.n):
                vals = dp[t-1] + logA[:, j]
                ptr[t, j] = int(np.argmax(vals))
                dp[t, j] = vals[ptr[t, j]] + logB[t, j]
        states = np.empty(T, dtype=int)
        states[-1] = int(np.argmax(dp[-1]))
        for t in range(T-2, -1, -1):
            states[t] = ptr[t+1, states[t+1]]
        return states

    def fit_bw(self, obs: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> "GaussianHMM":
        obs = np.asarray(obs, dtype=float)
        T = len(obs)
        for _ in range(max_iter):
            logB = self._log_emission(obs)
            logA = np.log(self.A + 1e-12)
            logpi = np.log(self.pi + 1e-12)
            # forward (log)
            alpha = np.zeros((T, self.n))
            alpha[0] = logpi + logB[0]
            for t in range(1, T):
                for j in range(self.n):
                    alpha[t, j] = logsumexp(alpha[t-1] + logA[:, j]) + logB[t, j]
            # backward (log)
            beta = np.zeros((T, self.n))
            beta[-1] = 0.0
            for t in range(T-2, -1, -1):
                for i in range(self.n):
                    beta[t, i] = logsumexp(logA[i] + logB[t+1] + beta[t+1])
            log_gamma = alpha + beta
            # normalize
            log_gamma = log_gamma - logsumexp(log_gamma, axis=1)[:, None]
            gamma = np.exp(log_gamma)
            # xi sums
            xi_sum = np.zeros((self.n, self.n))
            for t in range(T-1):
                mat = (alpha[t][:, None] + logA + logB[t+1][None, :] + beta[t+1][None, :])
                mat = mat - logsumexp(mat)
                xi_sum += np.exp(mat)
            # M-step
            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            for i in range(self.n):
                w = gamma[:, i]
                denom = w.sum()
                self.means[i] = (w * obs).sum() / denom
                if self.var_shared is None:
                    self.vars[i] = (w * (obs - self.means[i])**2).sum() / denom
            if self.var_shared is not None:
                var_tot = 0.0
                for i in range(self.n):
                    var_tot += (gamma[:, i] * (obs - self.means[i])**2).sum()
                self.vars[:] = var_tot / T
        return self

def logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    res = a_max + s
    if axis is not None:
        return np.squeeze(res, axis=axis)
    return res

def _example_hmm():
    rng = np.random.default_rng(4)
    n = 2
    true = GaussianHMM(n_states=n)
    true.pi = np.array([0.6, 0.4])
    true.A = np.array([[0.9, 0.1], [0.2, 0.8]])
    true.means = np.array([-1.0, 2.0])
    true.vars = np.array([0.5, 0.5])
    T = 300
    s = np.empty(T, dtype=int)
    obs = np.empty(T)
    s[0] = rng.choice(n, p=true.pi)
    obs[0] = rng.normal(true.means[s[0]], math.sqrt(true.vars[s[0]]))
    for t in range(1, T):
        s[t] = rng.choice(n, p=true.A[s[t-1]])
        obs[t] = rng.normal(true.means[s[t]], math.sqrt(true.vars[s[t]]))
    model = GaussianHMM(n_states=2)
    model.fit_bw(obs, max_iter=10)
    states = model.viterbi(obs)
    print("HMM states len:", len(states))

# ------------------------------
# 8) Bayesian Online Change Point Detection (Adams & MacKay) — simplified
# Problem:
#  - Implement BOCPD for Gaussian observations with known observation variance and conjugate prior on mean.
#
# Math (outline):
#  - Maintain run-length posterior P(r_t | x_{1:t}).
#  - Predictive probability p(x_t | r_{t-1}) using conjugate normal prior with parameter kappa, mu.
#  - Update run-length distribution via hazard function H: growth and changepoint branches.
#
# Implementation: simple version for readability (not optimized).
# ------------------------------
class BOCPD:
    def __init__(self, hazard: float = 1/200.0, mu0: float = 0.0, kappa0: float = 1.0):
        self.hazard = hazard
        self.mu0 = mu0
        self.kappa0 = kappa0

    def run(self, data: np.ndarray, sigma2_obs: float = 1.0) -> np.ndarray:
        T = len(data)
        R = np.zeros((T+1, T+1))
        R[0, 0] = 1.0
        mu_dict = {0: self.mu0}
        kappa_dict = {0: self.kappa0}
        for t in range(1, T+1):
            x = data[t-1]
            pred = np.zeros(t)
            for r in range(t):
                m = mu_dict[r]; k = kappa_dict[r]
                var_pred = sigma2_obs * (1 + 1.0 / k)
                pred[r] = stats.norm.pdf(x, loc=m, scale=math.sqrt(var_pred))
            # growth
            R[t, 1:t+1] = R[t-1, 0:t] * pred * (1 - self.hazard)
            # changepoint
            R[t, 0] = (R[t-1, :t] * pred).sum() * self.hazard
            # normalize
            s = R[t, :t+1].sum()
            if s > 0:
                R[t, :t+1] /= s
            # update sufficient stats
            new_mu = {}
            new_kappa = {}
            for r in range(t+1):
                if r == 0:
                    new_mu[r] = self.mu0
                    new_kappa[r] = self.kappa0
                else:
                    prev = r-1
                    prev_mu = mu_dict.get(prev, self.mu0)
                    prev_k = kappa_dict.get(prev, self.kappa0)
                    k_new = prev_k + 1
                    mu_new = (prev_k * prev_mu + x) / k_new
                    new_mu[r] = mu_new
                    new_kappa[r] = k_new
            mu_dict = new_mu; kappa_dict = new_kappa
        return R

def _example_bocpd():
    rng = np.random.default_rng(5)
    data = np.concatenate([rng.normal(0,1,100), rng.normal(3,1,100)])
    detector = BOCPD(hazard=1/100.0)
    R = detector.run(data)
    print("BOCPD shape:", R.shape)

# ------------------------------
# 9) Transaction-cost-aware portfolio optimization (quadratic + L1 prox)
# Problem:
#  - Solve approximately: max_w mu^T w - gamma/2 w^T Sigma w - c ||w - w0||_1
#  - Use coordinate descent and soft-thresholding prox for L1.
#
# Math idea:
#  - Equivalent to minimize 1/2 w^T (gamma Sigma) w - mu^T w + c ||w - w0||_1.
#  - For coordinate i, solve 1/2 a w_i^2 + b w_i + c |w_i - w0_i| where a = (gamma Sigma)_{ii} and b = gradient excluding i.
#  - Use soft-thresholding on shifted variable u = w_i - w0_i.
# ------------------------------
def soft_threshold(v: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(v) * np.maximum(0.0, np.abs(v) - lam)

def tc_portfolio(mu: np.ndarray, Sigma: np.ndarray, w0: np.ndarray, gamma: float = 1.0,
                 c: float = 1e-3, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    p = mu.size
    w = w0.copy().astype(float)
    G = gamma * Sigma
    diag = np.diag(G)
    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(p):
            g_i = G[i].dot(w) - G[i,i] * w[i] - mu[i]
            a = diag[i]
            if a == 0:
                continue
            z = - (g_i + a * w0[i]) / a
            u = soft_threshold(np.array([z]), c / a)[0]
            w[i] = u + w0[i]
        if np.linalg.norm(w - w_old) < tol:
            break
    return w

def _example_tc_portfolio():
    rng = np.random.default_rng(6)
    p = 10
    A = rng.normal(size=(p,p))
    Sigma = A.dot(A.T)
    mu = rng.normal(size=p)
    w0 = np.zeros(p)
    w = tc_portfolio(mu, Sigma, w0, gamma=1.0, c=1e-2)
    print("Optimized w shape:", w.shape)

# ------------------------------
# 10) Explainable ML: SHAP-like contributions for linear model and shallow tree
# Problem:
#  - For linear model: contributions per feature are coef_i * x_i; baseline intercept.
#  - For a small decision tree (depth <= 2), attribute path contributions deterministically (path decomposition).
#
# Implementation: functions linear_feature_contrib and tree_feature_contrib (toy exact path attribution).
# ------------------------------
def linear_feature_contrib(x: np.ndarray, coef: np.ndarray, intercept: float) -> Dict[str, float]:
    contribs = {f"f{i}": float(coef[i] * x[i]) for i in range(len(x))}
    contribs["baseline"] = float(intercept)
    contribs["prediction"] = float(intercept + coef.dot(x))
    return contribs

@dataclass
class SimpleTreeNode:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['SimpleTreeNode'] = None
    right: Optional['SimpleTreeNode'] = None
    value: Optional[float] = None  # leaf value

def tree_feature_contrib(x: np.ndarray, root: SimpleTreeNode) -> Dict[str, float]:
    contribs: Dict[str, float] = {}
    # baseline: use root average if leaf else 0. Here we use root.value if present else 0.
    baseline = root.value if root.value is not None else 0.0
    node = root
    value = baseline
    while node and node.value is None:
        f = node.feature; thr = node.threshold
        if f is None:
            break
        if x[f] <= thr:
            left_val = node.left.value if node.left and node.left.value is not None else 0.0
            delta = left_val - value
            contribs[f"f{f}"] = contribs.get(f"f{f}", 0.0) + float(delta)
            value = left_val
            node = node.left
        else:
            right_val = node.right.value if node.right and node.right.value is not None else 0.0
            delta = right_val - value
            contribs[f"f{f}"] = contribs.get(f"f{f}", 0.0) + float(delta)
            value = right_val
            node = node.right
    contribs["output"] = float(value)
    return contribs

def _example_feature_contrib():
    x = np.array([1.0, 2.0, 3.0])
    coef = np.array([0.5, -0.2, 0.1])
    intercept = 0.05
    print("Linear contributions:", linear_feature_contrib(x, coef, intercept))
    root = SimpleTreeNode(feature=0, threshold=0.5,
                          left=SimpleTreeNode(value=0.1),
                          right=SimpleTreeNode(feature=1, threshold=1.5,
                                               left=SimpleTreeNode(value=0.2),
                                               right=SimpleTreeNode(value=0.5)))
    print("Tree contributions:", tree_feature_contrib(x, root))

# ------------------------------
# Quick smoke-run function (runs small examples)
# ------------------------------
def run_smoke_examples():
    print("1) AR(1) estimators example:")
    _example_ar1()
    print("\n2) Pair trading example:")
    _example_pair()
    print("\n3) Online PCA example:")
    _example_online_pca()
    print("\n4) Ledoit-Wolf example:")
    _example_lw()
    print("\n5) Factor regressor example:")
    _example_factor()
    print("\n6) P2 quantile example:")
    _example_p2()
    print("\n7) HMM example:")
    _example_hmm()
    print("\n8) BOCPD example:")
    _example_bocpd()
    print("\n9) TC-aware portfolio example:")
    _example_tc_portfolio()
    print("\n10) Feature contributions example:")
    _example_feature_contrib()
    print("\nSmoke examples complete.")

if __name__ == "__main__":
    run_smoke_examples()

