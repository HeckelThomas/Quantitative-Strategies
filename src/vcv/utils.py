import numpy as np

def vcv_simulation(n: int=10, k: int=5, vol: float=0.2, seed: int=0) -> np.ndarray:
    """
    VCV random simulation using k factors and n idiosyncratic noises
    Random uniform volatility between 0.5*vol and 1.5*vol
    :param n: number of assets
    :param k: number of factors
    :param vol: average volatility
    :param seed: default seed
    :return: (n,n) random VCV
    """
    # Set random seed
    rng = np.random.default_rng(seed)
    # Build vcv
    beta_fact = rng.normal(size=(n,k))
    vcv_fact = beta_fact.dot(beta_fact.T)
    vcv_idio = np.identity(n)*np.diag(vcv_fact).mean()
    vcv = vcv_fact+vcv_idio
    # Adjust vol
    sigma = rng.uniform(low=0.5*vol, high= 1.5*vol, size=n)
    d = np.diag(sigma/np.sqrt(np.diag(vcv)))
    vcv = d.dot(vcv).dot(d)
    return vcv

def returns_simulation(vcv: np.ndarray, sr: float=0.3, cash: float=0.03, t: int=260, freq: int=260, seed: int=0) -> np.ndarray:
    """
    Returns simulations for n assets on t periods with a frequency freq per year
    :param vcv: VCV
    :param sr: Sharpe Ratio assumption
    :param cash: Cash assumption
    :param t: Number of periods to simulate
    :param freq: Number of periods per year
    :param seed: Default seed
    :return: Returns matrix
    """
    # Set random seed
    rng = np.random.default_rng(seed)
    # Simulate correlated noise with vcv
    n = len(vcv)
    eps = rng.normal(size=(t,n))
    L = np.linalg.cholesky(vcv/freq)
    eps = eps.dot(L.T)
    # Simulate returns with cash and sr assumptions
    returns = (cash + np.sqrt(np.diag(vcv))*sr)/freq + eps
    return returns
