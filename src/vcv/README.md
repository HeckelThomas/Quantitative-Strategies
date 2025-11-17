# Estimation — Covariance

VCV Estimation using Ledoit-Wolf

## Méthodes incluses
- `sample_cov` : empirical covariance
- `ledoit_wolf_shrinkage` : Ledoit-Wolf

## Fichiers
- `estimator.py` : estimation function
- `utils.py` : util functions

## Exemple of usage
```python
from src.vcv.estimator import *
from src.vcv.utils import *

vcv = vcv_simulation()
returns = returns_simulation(vcv)
vcv_lw, vol_lw, correl_lw, vcv_emp, vol_emp, correl_emp, delta = vcv_estimation(returns)
print("delta = ", delta)
print("vol_lw = ", vol_lw)

