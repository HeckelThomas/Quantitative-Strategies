
from src.vcv.estimator import *
from src.vcv.utils import *

def main():
    vcv = vcv_simulation()
    returns = returns_simulation(vcv)
    vcv_lw, vol_lw, correl_lw, vcv_emp, vol_emp, correl_emp, delta = vcv_estimation(returns)
    print("delta = ", delta)
    print("vol_lw = ", vol_lw)

if __name__ == "__main__":
    main()


