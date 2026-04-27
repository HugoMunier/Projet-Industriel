"""
garch_feature.py
----------------
Fonctions utilitaires pour le calcul de la volatilité conditionnelle GARCH(1,1),
utilisée comme variable d'entrée du classificateur LSTM de régimes de stress.
"""

import numpy as np
import pandas as pd
from arch import arch_model


def compute_garch_vol(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "skewt",
) -> pd.Series:
    """
    Ajuste un modèle GARCH(p, q) et retourne la série de volatilité conditionnelle.

    Les rendements sont multipliés par 100 avant l'estimation (pratique standard
    avec la librairie arch) puis ramenés à l'échelle originale avant le retour.

    Paramètres
    ----------
    returns : pd.Series
        Série de rendements logarithmiques (échelle décimale, ex. 0.01 pour 1%).
    p : int
        Ordre des termes ARCH. Par défaut 1.
    q : int
        Ordre des termes GARCH. Par défaut 1.
    dist : str
        Distribution des innovations passée à arch_model. Par défaut 'skewt'.

    Retourne
    --------
    pd.Series
        Volatilité conditionnelle à l'échelle décimale, indexée comme `returns`.
    """
    returns_scaled = returns * 100
    model = arch_model(returns_scaled, vol="Garch", p=p, q=q, dist=dist)
    result = model.fit(disp="off")
    cond_vol = result.conditional_volatility / 100
    return cond_vol


def get_garch_features(df: pd.DataFrame) -> pd.Series:
    """
    Calcule la volatilité conditionnelle GARCH(1,1) à partir d'un DataFrame de prix.

    Si la colonne 'LogRet' est absente, elle est calculée à la volée depuis 'Close'.
    Le résultat est réindexé sur l'index complet du DataFrame (NaN pour la première
    ligne où aucun rendement ne peut être calculé).

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame avec au minimum une colonne 'Close' et un DatetimeIndex.

    Retourne
    --------
    pd.Series
        Volatilité conditionnelle alignée sur df.index.
    """
    if "LogRet" not in df.columns:
        df = df.copy()
        df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))

    valid_returns = df["LogRet"].dropna()
    vol_series = compute_garch_vol(valid_returns)
    return vol_series.reindex(df.index)
