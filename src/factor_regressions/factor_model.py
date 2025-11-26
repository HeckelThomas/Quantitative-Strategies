import numpy as np

class factor_model:
    def __init__(self, alpha: float=0):
        self.intercepts: np.ndarray = None
        self.factors_exposures: np.ndarray = None
        self.alpha: float = alpha

    def fit(self, assets_xret: np.ndarray, factors_xret: np.ndarray) -> "factor_model":
        t, n = assets_xret.shape
        k = factors_xret.shape[1]
        intercepts = np.full((n, 1), np.nan)
        factors_exposures = np.full((n, k), np.nan)
        x = np.hstack((np.ones((t,1)),factors_xret))
        for j in range(n):
            coef_ = np.linalg.lstsq(x, assets_xret[:,j])[0]
            intercepts[j,:] = coef_[0]
            factors_exposures[j,:] = coef_[1:]
        factors_exposures_mean = factors_exposures.mean(axis=0)
        self.factors_exposures = (1-self.alpha)*factors_exposures + self.alpha*factors_exposures_mean
        self.intercepts = intercepts
        return self

    def predict(self, factors_xret: np.ndarray) -> np.ndarray:
        return self.intercepts.T + factors_xret.dot(self.factors_exposures.T)


