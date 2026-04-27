"""
conditional_volatility.py
--------------------------
Analyse économétrique de la volatilité conditionnelle sur les rendements du S&P 500.

Pipeline
--------
1. Chargement et nettoyage des données (tests de stationnarité ADF/KPSS)
2. Ingénierie des indicateurs techniques (RSI, MACD, ATR, vol. Parkinson, etc.)
3. Sélection du modèle de moyenne via Box-Jenkins (grille ARMA + test de Ljung-Box)
4. Sélection du modèle de variance : ARCH, GARCH, EGARCH, GJR-GARCH avec loi skew-t
5. Boucle de prévision hors-échantillon (OOS) glissante avec backtesting de la VaR
   (test de Kupiec POF + test d'indépendance de Christoffersen)

Utilisation
-----------
    python conditional_volatility.py --data chemin/vers/SP500.csv

Le fichier CSV doit contenir les colonnes : Date, Open, High, Low, Close, Volume.
"""

import argparse
import warnings
from math import log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import statsmodels.api as sm
from arch import arch_model
from arch.univariate.distribution import SkewStudent
from scipy.stats import chi2, norm, t
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from tqdm import tqdm

warnings.filterwarnings("ignore", category=InterpolationWarning)


# ---------------------------------------------------------------------------
# Fonctions auxiliaires pour la VaR
# ---------------------------------------------------------------------------

def skewt_quantile_arch(alpha: float, lam: float, nu: float) -> float:
    """
    Calcule le quantile d'ordre alpha de la loi de Student asymétrique (skew-t)
    telle que paramétrée par la librairie arch.

    Se replie sur une méthode de bissection, puis sur le quantile de la Student
    symétrique si tous les appels directs échouent.

    Paramètres
    ----------
    alpha : float
        Niveau de probabilité (ex. 0.01 pour la VaR à 1%).
    lam : float
        Paramètre d'asymétrie (lambda) issu du modèle arch ajusté.
    nu : float
        Degrés de liberté (eta/nu) issus du modèle arch ajusté.

    Retourne
    --------
    float
        Valeur du quantile.
    """
    d = SkewStudent()

    for args in [(lam, nu), ([lam, nu])]:
        try:
            return float(d.ppf(alpha, *args))
        except Exception:
            pass
        try:
            return float(SkewStudent.ppf(alpha, *args))
        except Exception:
            pass

    # Repli par bissection
    try:
        def cdf_x(x):
            try:
                return float(d.cdf(x, lam, nu))
            except Exception:
                return float(d.cdf(x, [lam, nu]))

        lo, hi = -20.0, 20.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if cdf_x(mid) < alpha:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    except Exception:
        pass

    return float(t.ppf(alpha, df=nu))


def kupiec_pof(
    returns: pd.Series,
    var_series: pd.Series,
    alpha: float = 0.01,
) -> dict:
    """
    Test du ratio de vraisemblance de Kupiec (1995) — Proportion of Failures (POF).

    Paramètres
    ----------
    returns : pd.Series
        Rendements logarithmiques réalisés.
    var_series : pd.Series
        VaR prédite (seuil négatif, même convention de signe que les rendements).
    alpha : float
        Niveau de confiance nominal de la VaR (ex. 0.01).

    Retourne
    --------
    dict
        Statistiques du test et p-value.
    """
    exceedances = (returns < var_series).astype(int).values
    T = len(exceedances)
    x = exceedances.sum()
    p_hat = x / T

    if p_hat in (0.0, 1.0):
        lr_pof, pval = 0.0, 1.0
    else:
        ll_h0 = x * log(alpha) + (T - x) * log(1 - alpha)
        ll_h1 = x * log(p_hat) + (T - x) * log(1 - p_hat)
        lr_pof = -2 * (ll_h0 - ll_h1)
        pval = 1 - chi2.cdf(lr_pof, df=1)

    return {
        "Exceptions": int(x),
        "Total": int(T),
        "Taux_observe": p_hat,
        "Taux_attendu": alpha,
        "LR_POF": lr_pof,
        "p_value": pval,
    }


def christoffersen_ind(
    returns: pd.Series,
    var_series: pd.Series,
) -> dict:
    """
    Test d'indépendance de Christoffersen (1998) pour le clustering des dépassements VaR.

    Paramètres
    ----------
    returns : pd.Series
        Rendements logarithmiques réalisés.
    var_series : pd.Series
        Seuil de VaR prédit.

    Retourne
    --------
    dict
        Comptes de transitions, probabilités conditionnelles, statistique LR et p-value.
    """
    indicator = (returns < var_series).astype(int).values
    T = len(indicator)

    n00 = n01 = n10 = n11 = 0
    for i in range(1, T):
        prev, curr = indicator[i - 1], indicator[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi  = (n01 + n11) / (n00 + n01 + n10 + n11) if T > 0 else 0.0

    def safe_log(p: float) -> float:
        eps = 1e-12
        return np.log(np.clip(p, eps, 1 - eps))

    ll_h0 = (
        (n00 + n01) * safe_log(1 - pi)
        + n01 * safe_log(pi)
        + (n10 + n11) * safe_log(1 - pi)
        + n11 * safe_log(pi)
    )
    ll_h1 = (
        n00 * safe_log(1 - pi0)
        + n01 * safe_log(pi0)
        + n10 * safe_log(1 - pi1)
        + n11 * safe_log(pi1)
    )

    lr_ind = -2 * (ll_h0 - ll_h1)
    pval = 1 - chi2.cdf(lr_ind, df=1)

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi0": pi0, "pi1": pi1, "pi": pi,
        "LR_IND": lr_ind, "p_value": pval,
    }


# ---------------------------------------------------------------------------
# Analyse principale
# ---------------------------------------------------------------------------

def run_analysis(data_path: str) -> None:
    """
    Exécute le pipeline complet de modélisation de la volatilité.

    Paramètres
    ----------
    data_path : str
        Chemin vers le fichier CSV du S&P 500 (colonnes : Date, Open, High,
        Low, Close, Volume).
    """

    # ------------------------------------------------------------------
    # 1. Chargement et nettoyage
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    print(df.head())

    plt.figure(figsize=(12, 6))
    plt.plot(df["Close"], label="S&P 500 (Clôture)", color="blue")
    plt.title("Évolution du S&P 500 (2000–2024)")
    plt.xlabel("Date")
    plt.ylabel("Niveau de l'indice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Jours de bourse manquants
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=df.index.min(), end_date=df.index.max())
    missing_dates = schedule.index.difference(df.index)
    print(f"\nJours de bourse manquants : {len(missing_dates)}")

    # Rendements logarithmiques
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df_r = df.dropna(subset=["LogReturn"])

    # ------------------------------------------------------------------
    # 2. Ingénierie des variables
    # ------------------------------------------------------------------
    # Volatilité historique (fenêtre glissante 20 jours)
    df["Volatility20d"] = df["LogReturn"].rolling(20).std()
    df["Volatility20d_ann"] = df["Volatility20d"] * np.sqrt(252)

    # RSI (14 jours)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / loss)

    # MACD (12-26, signal 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ATR (14 jours)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = true_range.rolling(14).mean()

    # Volatilité de Parkinson
    df["ParkinsonVol"] = np.sqrt(
        (1 / (4 * np.log(2)))
        * (np.log(df["High"] / df["Low"]) ** 2).rolling(20).mean()
        * 252
    )

    # ------------------------------------------------------------------
    # 3. Tests de stationnarité sur les rendements log
    # ------------------------------------------------------------------
    close_dif = np.log(df["Close"]).diff().dropna()

    adf = adfuller(close_dif)
    kpss_stat = kpss(close_dif, regression="c")
    print(f"\nADF p-value  : {adf[1]:.4f}")
    print(f"KPSS p-value : {kpss_stat[1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Sélection du modèle de moyenne (grille ARMA)
    # ------------------------------------------------------------------
    mean_results = []
    for p in range(6):
        for q in range(6):
            try:
                fit = sm.tsa.ARIMA(close_dif, order=(p, 0, q)).fit()
                lb = acorr_ljungbox(fit.resid, lags=[10, 20], return_df=True)
                mean_results.append({
                    "p": p, "q": q,
                    "AIC": fit.aic, "BIC": fit.bic,
                    "LB10_pvalue": lb["lb_pvalue"].iloc[0],
                    "LB20_pvalue": lb["lb_pvalue"].iloc[1],
                })
            except Exception:
                pass

    df_mean = pd.DataFrame(mean_results).sort_values(["BIC", "AIC"])
    print("\nMeilleurs candidats pour le modèle de moyenne :\n", df_mean.head(10).to_string())

    # Modèle de moyenne retenu : GARCH pur (moyenne nulle), validé dans le rapport
    resid_mean = close_dif.copy()

    # ------------------------------------------------------------------
    # 5. Sélection du modèle de volatilité
    # ------------------------------------------------------------------
    vol_configs = [
        ("ARCH",   1, 0, "ARCH(1)"),
        ("GARCH",  1, 1, "GARCH(1,1)"),
        ("EGARCH", 1, 1, "EGARCH(1,1)"),
    ]
    vol_results = []
    for vol_type, p, q, label in vol_configs:
        for dist in ("t", "skewt"):
            try:
                model = arch_model(
                    resid_mean * 100, mean="Zero",
                    vol=vol_type, p=p, q=q, dist=dist,
                )
                fit = model.fit(disp="off")
                std_r  = fit.std_resid.dropna()
                lb_r   = acorr_ljungbox(std_r,      lags=[10, 20], return_df=True)
                lb_r2  = acorr_ljungbox(std_r ** 2, lags=[10, 20], return_df=True)
                vol_results.append({
                    "Modèle": label, "Distribution": dist,
                    "AIC": fit.aic, "BIC": fit.bic,
                    "LB_résidus_p10":    lb_r["lb_pvalue"].iloc[0],
                    "LB_résidus_p20":    lb_r["lb_pvalue"].iloc[1],
                    "LB_résidus²_p10":   lb_r2["lb_pvalue"].iloc[0],
                    "LB_résidus²_p20":   lb_r2["lb_pvalue"].iloc[1],
                    "Valide": (lb_r["lb_pvalue"].min() > 0.05
                               and lb_r2["lb_pvalue"].min() > 0.05),
                })
            except Exception:
                pass

    df_vol = pd.DataFrame(vol_results).sort_values("BIC")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\nSélection du modèle de volatilité :\n", df_vol.to_string())

    # ------------------------------------------------------------------
    # 6. Prévision OOS glissante — GARCH(1,1), EGARCH, GJR-GARCH
    # ------------------------------------------------------------------
    _run_oos_forecast(close_dif, model_type="GARCH",    label="GARCH(1,1)")
    _run_oos_forecast(close_dif, model_type="EGARCH",   label="EGARCH(1,1)")
    _run_oos_forecast(close_dif, model_type="GJRGARCH", label="GJR-GARCH(1,1)")


def _run_oos_forecast(
    close_dif: pd.Series,
    model_type: str,
    label: str,
    start_test: str = "2010-01-01",
    end_test: str = "2024-12-31",
    train_years: int = 10,
    refit_freq: int = 20,
    alpha: float = 0.01,
) -> pd.DataFrame:
    """
    Prévision de volatilité hors-échantillon glissante avec backtesting de la VaR.

    Paramètres
    ----------
    close_dif : pd.Series
        Série de rendements log (historique complet).
    model_type : str
        L'un des modèles suivants : 'GARCH', 'EGARCH', 'GJRGARCH'.
    label : str
        Nom lisible du modèle pour les titres de graphiques.
    start_test : str
        Date de début de la période OOS.
    end_test : str
        Date de fin de la période OOS.
    train_years : int
        Longueur de la fenêtre glissante d'entraînement (en années).
    refit_freq : int
        Nombre de pas OOS entre deux ré-estimations du modèle.
    alpha : float
        Niveau de confiance de la VaR.

    Retourne
    --------
    pd.DataFrame
        Résultats OOS avec les séries de volatilité prédite/réalisée et de VaR.
    """
    vol_kwargs = {
        "GARCH":    {"vol": "GARCH", "p": 1, "q": 1},
        "EGARCH":   {"vol": "EGARCH", "p": 1, "q": 1},
        "GJRGARCH": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    }[model_type]

    test_dates = close_dif.loc[start_test:end_test].index
    pred_var, pred_sigma, pred_sigma_ann = [], [], []
    real_var, returns_list, var_skewt_list = [], [], []

    fit_oos = None
    q_skewt = None

    for i, date in tqdm(enumerate(test_dates), total=len(test_dates),
                        desc=f"OOS {label}"):
        if i % refit_freq == 0 or fit_oos is None:
            train_start = date - pd.DateOffset(years=train_years)
            train = close_dif.loc[train_start: date - pd.Timedelta(days=1)]

            model = arch_model(
                train * 100, mean="Zero", dist="skewt", **vol_kwargs
            )
            fit_oos = model.fit(disp="off")

            nu  = float(fit_oos.params.get("eta", fit_oos.params.get("nu")))
            lam = float(fit_oos.params["lambda"])
            q_skewt = skewt_quantile_arch(alpha, lam, nu)

        forecast = fit_oos.forecast(horizon=1, reindex=False)
        var_pred   = forecast.variance.values[-1, 0] / 100 ** 2
        sigma_pred = np.sqrt(var_pred)

        pred_var.append(var_pred)
        pred_sigma.append(sigma_pred)
        pred_sigma_ann.append(sigma_pred * np.sqrt(252))
        real_var.append(close_dif.loc[date] ** 2)
        returns_list.append(close_dif.loc[date])
        var_skewt_list.append(q_skewt * sigma_pred)

    oos = pd.DataFrame({
        "Pred_Var":       pred_var,
        "Pred_Sigma":     pred_sigma,
        "Pred_Sigma_Ann": pred_sigma_ann,
        "Real_Var":       real_var,
        "Rendement":      returns_list,
        "VaR_1pct":       var_skewt_list,
    }, index=test_dates[: len(pred_sigma)])
    oos["Real_Sigma"]     = np.sqrt(oos["Real_Var"])
    oos["Real_Sigma_Ann"] = oos["Real_Sigma"] * np.sqrt(252)

    # Métriques de perte
    oos["QLIKE"] = np.log(oos["Pred_Var"]) + oos["Real_Var"] / oos["Pred_Var"]
    oos["MSE"]   = (oos["Real_Var"] - oos["Pred_Var"]) ** 2
    oos["MAE"]   = (oos["Real_Sigma"] - oos["Pred_Sigma"]).abs()

    print(f"\n--- Métriques OOS {label} ({start_test}–{end_test}) ---")
    print(oos[["QLIKE", "MSE", "MAE"]].mean().to_string())

    # Tests de backtesting VaR
    res_pof = kupiec_pof(oos["Rendement"], oos["VaR_1pct"], alpha=alpha)
    res_ind = christoffersen_ind(oos["Rendement"], oos["VaR_1pct"])

    print(f"\nTest de Kupiec POF ({label}) :")
    for k, v in res_pof.items():
        print(f"  {k:<20s}: {v:.4f}" if isinstance(v, float) else f"  {k:<20s}: {v}")

    print(f"\nTest d'indépendance de Christoffersen ({label}) :")
    for k, v in res_ind.items():
        print(f"  {k:<20s}: {v:.4f}" if isinstance(v, float) else f"  {k:<20s}: {v}")

    # Graphique
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(oos.index, oos["Pred_Sigma_Ann"], label="Volatilité prédite (annualisée)", color="red")
    ax.plot(oos.index, oos["Real_Sigma_Ann"], label="Volatilité réalisée (annualisée)", color="blue", alpha=0.7)
    ax.set_title(f"Volatilité conditionnelle — {label}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return oos


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse de la volatilité conditionnelle du S&P 500 (famille GARCH)"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Chemin vers le fichier CSV SP500 (colonnes : Date, Open, High, Low, Close, Volume)",
    )
    args = parser.parse_args()
    run_analysis(args.data)
