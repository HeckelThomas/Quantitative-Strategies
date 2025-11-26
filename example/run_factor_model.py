import numpy as np
from src.factor_regressions.factor_model import factor_model
from src.factor_regressions.utils import simulate_factor_and_asset_returns

def main():
    # Asset return simulation
    assets_ret, assets_xret, factors_xret, assets_beta = simulate_factor_and_asset_returns()
    assets_xret_mean = assets_xret.mean(axis=0)*260
    assets_xret_vol = assets_xret.std(axis=0) * np.sqrt(260)
    assets_xret_sr = assets_xret_mean/assets_xret_vol
    print("Mean excess returns:\n", assets_xret_mean.round(2))
    print("Volatilities:\n", assets_xret_vol.round(2))
    print("Sharpe Ratio:\n", assets_xret_sr.round(2))

    fm = factor_model(0.0)
    fm.fit(assets_xret, factors_xret)

    n, k =  assets_beta.shape
    compare_beta = np.full((2, n, k), np.nan)
    compare_beta[0] = assets_beta
    compare_beta[1] = fm.factors_exposures
    for i in range(n):
        print(f"True and Estimated factor exposures for asset {i}:\n", compare_beta[:,i,:].round(2))

    # Forecast of returns via factor model
    assets_xret_estimated = fm.predict(factors_xret)
    delta = assets_xret-assets_xret_estimated
    print("TE between assets xret and forecasted xret (likely to be close to idiosyncratic vol)\n", (delta.std(axis=0)*np.sqrt(260)).round(3))

if __name__=="__main__":
    main()
