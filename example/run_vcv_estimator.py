
from src.vcv.estimator import *
from src.vcv.utils import *

def main():
    vcv = vcv_simulation()
    returns = returns_simulation(vcv)
    vol, correl, delta = vcv_estimation(returns)
    print("\ndelta = ", delta.round(2), "\n")
    print("vol = \n", vol.round(3), "\n")
    print("correl upper=LW, lower=Emp\n", correl.round(2))

if __name__ == "__main__":
    main()


