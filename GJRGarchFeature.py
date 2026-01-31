import numpy as np
import pandas as pd
from arch import arch_model

def compute_gjrgarch_vol(returns, p=1, q=1, dist='skewt'):

    returns_scaled = returns * 100
    am = arch_model(returns_scaled, vol='Garch', p=p, o=1, q=q, dist=dist)
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility / 100
    
    return cond_vol

def get_gjrgarch_features(df):

    if "LogRet" not in df.columns:
        df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
    valid_returns = df["LogRet"].dropna()
    vol_series = compute_gjrgarch_vol(valid_returns)
    
    return vol_series.reindex(df.index)