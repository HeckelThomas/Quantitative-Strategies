# Factor Regressions

Time series factor regressions

## Méthodes incluses
- `fit` : time series estimation with shrinkage 
- `predict` : predict from factor exposures
- `simulate_factor_and_asset_returns` : simulate returns

## Fichiers
- `factor_model.py` : class with methods to estimate factor exposures and predict from them
- `utils.py` : util functions

## Exemple of usage
```python
from src.vcv.estimator import *
from src.vcv.utils import *

vcv = vcv_simulation()
returns = returns_simulation(vcv)
vol, correl, delta = vcv_estimation(returns)
print("\ndelta = ", delta.round(2), "\n")
print("vol = \n", vol.round(3), "\n")
print("correl upper=LW, lower=Emp\n", correl.round(2))

