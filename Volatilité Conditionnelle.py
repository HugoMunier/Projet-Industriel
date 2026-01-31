import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
import numpy as np
import warnings
import statsmodels.api as sm
from scipy.stats import norm, chi2, t
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from tqdm import tqdm
from math import log
from arch.univariate.distribution import SkewStudent
import scipy.stats as stats


# %%

def skewt_quantile_arch(alpha: float, lam: float, nu: float) -> float:
    
    d = SkewStudent()
    
    try:
        return float(d.ppf(alpha, lam, nu))
    except Exception:
        pass
    try:
        return float(d.ppf(alpha, [lam, nu]))
    except Exception:
        pass

    try:
        return float(SkewStudent.ppf(alpha, lam, nu))
    except Exception:
        pass
    try:
        return float(SkewStudent.ppf(alpha, [lam, nu]))
    except Exception:
        pass
    
    try:
        def cdf_x(x):
            try:
                return float(d.cdf(x, lam, nu))
            except Exception:
                return float(d.cdf(x, [lam, nu]))
        lo, hi = -20.0, 20.0
        for _ in range(80):  # bisection
            mid = 0.5*(lo+hi)
            if cdf_x(mid) < alpha:
                lo = mid
            else:
                hi = mid
        return 0.5*(lo+hi)
    except Exception:
        pass

    return float(t.ppf(alpha, df=nu))

def kupiec_pof(returns, var_series, alpha=0.01):
    I = (returns < var_series).astype(int).values
    T = len(I)
    x = I.sum()
    p_hat = x / T
    
    if p_hat == 0 or p_hat == 1:
        LR_pof = 0.0
        pval = 1.0
    else:
        ll_H0 = x*log(alpha) + (T-x)*log(1-alpha)
        ll_H1 = x*log(p_hat) + (T-x)*log(1-p_hat)
        LR_pof = -2*(ll_H0 - ll_H1)
        pval = 1 - chi2.cdf(LR_pof, df=1)
    
    return {
        "Exceptions": int(x),
        "Total": int(T),
        "Taux_observe": p_hat,
        "Taux_attendu": alpha,
        "LR_POF": LR_pof,
        "p_value": pval
    }

def christoffersen_ind(returns, var_series):
    I = (returns < var_series).astype(int).values  # exceptions (0/1)
    T = len(I)
    
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if I[t-1] == 0 and I[t] == 0: n00 += 1
        elif I[t-1] == 0 and I[t] == 1: n01 += 1
        elif I[t-1] == 1 and I[t] == 0: n10 += 1
        else: n11 += 1
        
    pi0 = n01 / (n00 + n01) if (n00+n01)>0 else 0
    pi1 = n11 / (n10 + n11) if (n10+n11)>0 else 0
    pi  = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00+n01+n10+n11)>0 else 0
    
    def safe_log(p):
        eps = 1e-12
        return np.log(max(min(p,1-eps), eps))
    
    ll_H0 = (n00+n01)*safe_log(1-pi) + n01*safe_log(pi) \
      + (n10+n11)*safe_log(1-pi) + n11*safe_log(pi)

    ll_H1 = n00*safe_log(1-pi0) + n01*safe_log(pi0) \
      + n10*safe_log(1-pi1) + n11*safe_log(pi1)

    LR_ind = -2*(ll_H0 - ll_H1)
    pval = 1 - chi2.cdf(LR_ind, df=1)

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi0": pi0, "pi1": pi1, "pi": pi,
        "LR_IND": LR_ind, "p_value": pval
    }

# %% Messages d'avis

warnings.filterwarnings("ignore", category=InterpolationWarning)

# %% Collecte de données & Première visualisation

df = pd.read_csv(
    "C:\\Users\\hugom\\OneDrive - Aescra Emlyon Business School\\Mines de Saint-Etienne\\3A\\Projet Indus\\Robo advisor\\Collecte de données\\SP500.csv",
    parse_dates=["Date"]
)

df.set_index("Date", inplace=True) # Mettre la colonne Date comme index
df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0) #Nettoyer la colonne Volume (remplacer "error" par 0)

print(df.head())

plt.figure(figsize=(12,6))
plt.plot(df["Close"], label="S&P 500 (Close)", color="blue")

plt.title("Évolution du S&P 500 (2000–2024)")
plt.xlabel("Date")
plt.ylabel("Indice (points)")
plt.legend()
plt.grid(True)
plt.show()


# %% Nettoyage du set de data

## Aperçu global
print("="*30, "APERCU", "="*30)
print(df.head())
print("\n", "="*30, "INFO", "="*30)
print(df.info())
print("\n", "="*30, "DESCRIPTIVES", "="*30)
print(df.describe())

## Valeurs manquantes
print("\n", "="*30, "VALEURS MANQUANTES", "="*30)
print(df.isnull().sum())
print("\n", "="*30, "PROPORTION DE VALEURS MANQUANTES", "="*30)
print(df.isnull().mean())

## Doublons
print("\n", "="*30, "DOUBLONS (date)", "="*30)
print(df.index.duplicated().sum())

## Dates manquantes
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=df.index.min(), end_date=df.index.max())
trading_days = schedule.index

missing_dates = trading_days.difference(df.index)

print("\n", "="*30, "DATES", "="*30)
print("Nombre de dates manquantes :", len(missing_dates))

## Outliers / Rendements log
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
df_r = df.dropna(subset=["LogReturn"])
# Après avoir calculé les rendements, on supprime la première ligne (NaN).
# C’est une pratique courante en économétrie/finance car le 1er jour n’a pas de rendement (pas de jour précédent).
print(df_r[["Close", "LogReturn"]].head())

df_r["LogReturn"].hist(bins=100)
plt.title("Distribution des rendements log")
plt.show()

df_r[["LogReturn"]].boxplot()
plt.show()
# On utilise les rendements plutôt que les prix car ce sont les variations qui révèlent les chocs de marché (outliers)
# On prend le log pour que les rendements soient plus faciles à comparer dans le temps et mieux adaptés aux calculs

data = df_r["LogReturn"]
mu, sigma = data.mean(), data.std()

plt.hist(data, bins=100, density=True, alpha=0.6, color="blue", label="Rendements réels")

x = np.linspace(data.min(), data.max(), 1000)

plt.plot(x, norm.pdf(x, mu, sigma), "r", linewidth=0.3, label="Loi normale")
plt.title("Distribution des rendements log vs loi normale")
plt.xlabel("Rendement log")
plt.ylabel("Densité")


params = stats.nct.fit(data) # Exemple avec non-central t, souvent proche

# Tracer la Skew-t par dessus l'histogramme
plt.plot(x, stats.nct.pdf(x, *params), "b", linewidth=0.5, label="Orientation : Skew-t")
plt.legend()
plt.show()

(osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=None)

plt.figure(figsize=(7, 7))

# On trace les points en PREMIER (zorder=1)
# On garde le bleu-violet pastel avec une opacité élevée pour qu'ils soient bien présents
plt.scatter(osm, osr, color="#95a5f5", alpha=0.8, 
            linewidth=0.3, s=40, label="Données réelles")

# On trace la ligne en DERNIER (zorder=2) pour qu'elle passe PAR-DESSUS les points
# On utilise le rouge corail doux
plt.plot(osm, slope * osm + intercept, color="#ef7a7a", linewidth=3, 
         label="Loi normale")

# Personnalisation
plt.title("Q-Q Plot : Analyse de la Normalité", fontsize=13, pad=15)
plt.xlabel("Quantiles théoriques", fontsize=11)
plt.ylabel("Quantiles des données", fontsize=11)

# Grille discrète en arrière-plan (zorder=0)
plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

plt.legend()
plt.show()
# %% Construction de features (rendements, volatilitén indicateurs techniques )

# Volatilité historique 
df["Volatility20d"] = df["LogReturn"].rolling(window=20).std()
df["Volatility20d_ann"] = df["Volatility20d"] * np.sqrt(252)

print("\n", "="*30, "VOLATILITE HISTORIQUE", "="*30)
print(df[["Volatility20d", "Volatility20d_ann"]].head(30))

plt.figure(figsize=(12,6))
df["Volatility20d_ann"].plot(color="red", label="Volatilité 20j annualisée")
plt.title("Volatilité historique (S&P500)")
plt.legend()
plt.show()

# Moyennes mobiles
df["SMA20"] = df["Close"].rolling(window=20).mean()
df["SMA50"] = df["Close"].rolling(window=50).mean()
df["SMA200"] = df["Close"].rolling(window=200).mean()

plt.figure(figsize=(12,5))
df["Close"].plot(label="Close", alpha=0.7, color="blue")
df["SMA20"].plot(label="SMA20 (court terme)", color="orange")
plt.title("S&P500 - Moyenne mobile 20 jours (court terme)")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
df["Close"].plot(label="Close", alpha=0.7, color="blue")
df["SMA50"].plot(label="SMA50 (moyen terme)", color="green")
plt.title("S&P500 - Moyenne mobile 50 jours (moyen terme)")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
df["Close"].plot(label="Close", alpha=0.7, color="blue")
df["SMA200"].plot(label="SMA200 (long terme)", color="red")
plt.title("S&P500 - Moyenne mobile 200 jours (long terme)")
plt.legend()
plt.show()

# EMA (exponential moving average)
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

plt.figure(figsize=(12,5))
df["Close"].plot(label="Close", alpha=0.7, color="blue")
df["EMA20"].plot(label="EMA20 (court terme)", color="purple")
plt.title("S&P500 - Moyenne mobile exponentielle 20 jours (EMA)")
plt.legend()
plt.show()

# Average True Range
high_low = df["High"] - df["Low"]
high_close = np.abs(df["High"] - df["Close"].shift(1))
low_close = np.abs(df["Low"] - df["Close"].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["ATR14"] = true_range.rolling(window=14).mean()

plt.figure(figsize=(12,5))
df["ATR14"].plot(color="red", label="ATR (14 jours)")
plt.title("S&P500 - Average True Range (ATR14)")
plt.ylabel("ATR")
plt.legend()
plt.show()

# Parkinson volatility (High/Low)
df["ParkinsonVol"] = (1/(4*np.log(2))) * ((np.log(df["High"]/df["Low"]))**2)
df["ParkinsonVol"] = np.sqrt(df["ParkinsonVol"].rolling(window=20).mean()) * np.sqrt(252)

plt.figure(figsize=(12,6))
df["Volatility20d_ann"].plot(label="Volatilité 20j annualisée")
df["ParkinsonVol"].plot(label="Volatilité Parkinson (20j)")
plt.title("Volatilité historique annualisée")
plt.legend()
plt.show()

# RSI (Relative Strengh Index, 14 jours)
window = 14
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=window, min_periods=window).mean()
avg_loss = loss.rolling(window=window, min_periods=window).mean()
rs = avg_gain / avg_loss
df["RSI14"] = 100 - (100 / (1 + rs))

plt.figure(figsize=(12,4))
df["RSI14"].plot(color="purple")
plt.axhline(70, color="red", linestyle="--", alpha=0.7)
plt.axhline(30, color="green", linestyle="--", alpha=0.7)
plt.title("RSI (14 jours)")
plt.show()

# MACD (12-26 EMA, signal 9)
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

plt.figure(figsize=(12,4))
df["MACD"].plot(label="MACD", color="blue")
df["MACD_signal"].plot(label="Signal (9j)", color="red")
plt.axhline(0, color="black", linestyle="--", alpha=0.7)
plt.title("MACD")
plt.legend()
plt.show()

# Skewness & Kurtosis (rolling 252 jours = 1 an)
df["Skewness252"] = df["LogReturn"].rolling(window=252).skew()
df["Kurtosis252"] = df["LogReturn"].rolling(window=252).kurt()

plt.figure(figsize=(12,4))
df["Skewness252"].plot(color="blue", label="Skewness (252 jours)")
plt.axhline(0, color="black", linestyle="--", alpha=0.7)
plt.title("Asymétrie (Skewness) glissante sur 1 an")
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
df["Kurtosis252"].plot(color="red", label="Kurtosis (252 jours)")
plt.axhline(3, color="black", linestyle="--", alpha=0.7, label="Loi normale (k=3)")
plt.title("Aplatissement (Kurtosis) glissant sur 1 an")
plt.legend()
plt.show()


# %% Modélisation
close = df["Close"]

plt.figure(figsize=(10,4))
plt.plot(close)
plt.title("Série des prix de clôture S&P500")
plt.ylabel("Close")
plt.show()

# ---------------ADF & KPSS---------------

adf_result = adfuller(close)
print("\n","="*30, "ADF", "="*30)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

kpss_result = kpss(close, regression="c")
print("\n","="*30, "KPSS", "="*30)
print("KPSS Statistic:", kpss_result[0])
print("p-value:", kpss_result[1])

# ---------------Différenciation---------------

log_close = np.log(close)
close_dif = log_close.diff().dropna()

plt.figure(figsize=(10,4))
plt.plot(close_dif)
plt.title("Série différenciée log")
plt.ylabel("Log-close")
plt.show()


adf_result_dif = adfuller(close_dif)
print("\n","="*30, "ADF (close_diff)", "="*30)
print("ADF Statistic:", adf_result_dif[0])
print("p-value:", adf_result_dif[1])

kpss_result_dif = kpss(close_dif, regression="c")
print("\n","="*30, "KPSS (close_diff)", "="*30)
print("KPSS Statistic:", kpss_result_dif[0])
print("p-value:", kpss_result_dif[1])

# ---------------ACF & PACF---------------

acf_vals = acf(close_dif, nlags=30)
pacf_vals = pacf(close_dif, nlags=30, method="ywm")
lags = np.arange(len(acf_vals))
conf = 1.96/np.sqrt(len(close_dif))

plt.figure(figsize=(10,4))
plt.bar(lags, acf_vals, width=0.15, color="steelblue", edgecolor="steelblue")  # width plus petit = barres plus fines
plt.axhline(y=0, color="black", linewidth=0.8)
plt.axhline(y=conf, color="red", linestyle="--", linewidth=0.7)
plt.axhline(y=-conf, color="red", linestyle="--", linewidth=0.7)
plt.title("ACF (barres fines)", fontsize=14)
plt.show()

plt.figure(figsize=(10,4))
plt.bar(lags, pacf_vals, width=0.15, color="steelblue", edgecolor="steelblue")
plt.axhline(y=0, color="black", linewidth=0.8)
plt.axhline(y=conf, color="red", linestyle="--", linewidth=0.7)
plt.axhline(y=-conf, color="red", linestyle="--", linewidth=0.7)
plt.title("PACF (barres fines)", fontsize=14)
plt.show()

# %% Première méthode (à la main)
# ---------------AR(1)---------------

AR1 = sm.tsa.ARIMA(close_dif, order=(1,0,0))
AR1_fit = AR1.fit()

# ---------------MA(1)---------------

MA1 = sm.tsa.ARIMA(close_dif, order=(0,0,1)) 
MA1_fit = MA1.fit()

# ---------------ARMA(1,1)---------------

ARMA11 = sm.tsa.ARIMA(close_dif, order=(1,0,1))
ARMA11_fit = ARMA11.fit()

# ---------------Modèle à moyenne nulle---------------

ARMA00 = sm.tsa.ARIMA(close_dif, order=(0,0,0))
ARMA00_fit = ARMA00.fit()

# ---------------AIC & BIC---------------

results = {
    "AR(1)": {"AIC": AR1_fit.aic, "BIC": AR1_fit.bic},
    "MA(1)": {"AIC": MA1_fit.aic, "BIC": MA1_fit.bic},
    "ARMA(1,1)": {"AIC": ARMA11_fit.aic, "BIC": ARMA11_fit.bic},
    "ARMA(0,0)": {"AIC": ARMA00_fit.aic, "BIC": ARMA00_fit.bic},
}

criteria = pd.DataFrame(results).T
print(criteria)

criteria.plot(kind="bar")
plt.title("Comparaison AIC et BIC des modèles")
plt.ylabel("Valeur du critère")
plt.xticks(rotation=0)
plt.show()

criteria_norm = criteria - criteria.min()
print(criteria_norm)

criteria_norm.plot(kind="bar")
plt.title("Écarts relatifs AIC et BIC par rapport au meilleur modèle")
plt.ylabel("Écart au minimum")
plt.xticks(rotation=0)
plt.show()

# ---------------Résidus---------------

res_AR1 = AR1_fit.resid
res_MA1 = MA1_fit.resid
res_ARMA11 = ARMA11_fit.resid
res_ARMA00 = ARMA00_fit.resid

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

# AR(1)
axes[0,0].hist(res_AR1, bins=30, alpha=0.7, color="steelblue")
axes[0,0].set_title("Résidus AR(1)")

# MA(1)
axes[0,1].hist(res_MA1, bins=30, alpha=0.7, color="orange")
axes[0,1].set_title("Résidus MA(1)")

# ARMA(1,1)
axes[1,0].hist(res_ARMA11, bins=30, alpha=0.7, color="green")
axes[1,0].set_title("Résidus ARMA(1,1)")

# ARMA(0,0)
axes[1,1].hist(res_ARMA00, bins=30, alpha=0.7, color="red")
axes[1,1].set_title("Résidus ARMA(0,0)")

plt.tight_layout()
plt.show()

# ---------------Ljung-Box---------------

lb_AR1 = acorr_ljungbox(res_AR1, lags=[10, 20], return_df=True)
lb_MA1 = acorr_ljungbox(res_MA1, lags=[10, 20], return_df=True)
lb_ARMA11 = acorr_ljungbox(res_ARMA11, lags=[10, 20], return_df=True)
lb_ARMA00 = acorr_ljungbox(res_ARMA00, lags=[10, 20], return_df=True)

print("\n", "="*30,"Ljung-Box AR(1)", "="*30)
print(lb_AR1, "\n")
print("="*30,"Ljung-Box MA(1)", "="*30)
print(lb_MA1, "\n")
print("="*30,"Ljung-Box ARMA(1,1)", "="*30)
print(lb_ARMA11, "\n")
print("="*30,"Ljung-Box ARMA(0,0)", "="*30)
print(lb_ARMA00, "\n")

# ---------------ARCH LM---------------

arch_AR1 = het_arch(res_AR1, nlags=10)
arch_MA1 = het_arch(res_MA1, nlags=10)
arch_ARMA11 = het_arch(res_ARMA11, nlags=10)
arch_ARMA00 = het_arch(res_ARMA00, nlags=10)

print("\n","="*30,"ARCH LM AR(1)", "="*30)
print("LM stat:", arch_AR1[0], "p-value:", arch_AR1[1])
print("\n","="*30,"ARCH LM MA(1)", "="*30)
print("LM stat:", arch_MA1[0], "p-value:", arch_MA1[1])
print("\n","="*30,"ARCH LM ARMA(1,1)", "="*30)
print("LM stat:", arch_ARMA11[0], "p-value:", arch_ARMA11[1])
print("\n","="*30,"ARCH LM ARMA(0,0)", "="*30)
print("LM stat:", arch_ARMA00[0], "p-value:", arch_ARMA00[1])

# ---------------Rescaler les rendements---------------

close_dif_scaled = 100*close_dif

# ---------------ARMA(0,0)-ARCH(1)---------------

arch1 = arch_model(close_dif_scaled, vol='ARCH', p=1, mean='Zero', dist='normal')
arch1_fit = arch1.fit(disp='off')


# ---------------ARMA(0,0)-GARCH(1,1)---------------

garch11 = arch_model(close_dif_scaled, vol='GARCH', p=1, q=1, mean='Zero', dist='normal')
garch11_fit = garch11.fit(disp='off')

# ---------------Comparaison AIC & BIC---------------

crit = pd.DataFrame({
    "ARCH(1)": [arch1_fit.aic, arch1_fit.bic],
    "GARCH(1,1)": [garch11_fit.aic, garch11_fit.bic]
}, index=["AIC", "BIC"])
print("\nComparaison critères :\n", crit)

# ---------------Variance conditionnelle estimée---------------

arch1_var = pd.Series(arch1_fit.conditional_volatility/100, index=close_dif.index, name="ARCH_vol")
garch11_var = pd.Series(garch11_fit.conditional_volatility/100, index=close_dif.index, name="GARCH_vol")

plt.figure(figsize=(12,6))
plt.plot(close_dif, color="grey", alpha=0.6, label="Rendements")
plt.plot(arch1_var, color="blue", label="ARCH(1) - Volatilité")
plt.plot(garch11_var, color="red", label="GARCH(1,1) - Volatilité")
plt.title("Volatilité conditionnelle estimée (ARCH vs GARCH)")
plt.legend()
plt.show()

# ---------------Résidus ARMA(0,0)-GARCH(1,1) & ARMA(0,0)-ARCH(1,1)---------------

resid_arch = arch1_fit.resid
resid_garch = garch11_fit.resid

std_resid_arch = arch1_fit.std_resid
std_resid_garch = garch11_fit.std_resid

plt.figure(figsize=(10,4))
plt.plot(std_resid_garch, color="steelblue")
plt.title("Résidus standardisés GARCH(1,1)")
plt.show()

lb_arch = acorr_ljungbox(std_resid_arch, lags=[10,20], return_df=True)
print("\n","="*30,"Ljung-Box ARCH(1) Résidus standardisés","="*30)
print(lb_arch)

lb_garch = acorr_ljungbox(std_resid_garch, lags=[10,20], return_df=True)
print("\n","="*30,"Ljung-Box GARCH(1,1) Résidus standardisés","="*30)
print(lb_garch)

# ---------------AR(1)-ARCH(1)---------------

AR1_ARCH1 = arch_model(res_AR1 * 100, vol='ARCH', p=1, mean='Zero', dist='normal')
AR1_ARCH1_fit = AR1_ARCH1.fit(disp='off')

# ---------------AR(1)-GARCH(1,1)---------------

AR1_GARCH11 = arch_model(res_AR1 * 100, vol='GARCH', p=1, q=1, mean='Zero', dist='normal')
AR1_GARCH11_fit = AR1_GARCH11.fit(disp='off')

# ----------------- Comparaison AIC & BIC -----------------

crit = pd.DataFrame({
    "AR(1)-ARCH(1)": [AR1_ARCH1_fit.aic, AR1_ARCH1_fit.bic],
    "AR(1)-GARCH(1,1)": [AR1_GARCH11_fit.aic, AR1_GARCH11_fit.bic]
}, index=["AIC", "BIC"])
print("\nComparaison critères :\n", crit)

# ----------------- Variance conditionnelle estimée -----------------

ar1_arch1_var = pd.Series(AR1_ARCH1_fit.conditional_volatility / 100, 
                          index=AR1_ARCH1_fit.resid.index)

ar1_garch11_var = pd.Series(AR1_GARCH11_fit.conditional_volatility / 100, 
                            index=AR1_GARCH11_fit.resid.index)

plt.figure(figsize=(12,6))
plt.plot(close_dif.index[1:], close_dif.iloc[1:], color="grey", alpha=0.6, label="Rendements")
plt.plot(ar1_arch1_var, color="blue", label="AR(1)-ARCH(1) - Volatilité")
plt.plot(ar1_garch11_var, color="red", label="AR(1)-GARCH(1,1) - Volatilité")
plt.title("Volatilité conditionnelle estimée (AR(1)-ARCH vs AR(1)-GARCH)")
plt.legend()
plt.show()

# ----------------- Résidus standardisés -----------------

std_resid_arch = AR1_ARCH1_fit.std_resid
std_resid_garch = AR1_GARCH11_fit.std_resid

plt.figure(figsize=(10,4))
plt.plot(std_resid_garch, color="steelblue")
plt.title("Résidus standardisés AR(1)-GARCH(1,1)")
plt.show()

# ----------------- Ljung-Box sur résidus standardisés -----------------
lb_arch = acorr_ljungbox(std_resid_arch, lags=[10,20], return_df=True)
print("\n","="*30,"Ljung-Box AR(1)-ARCH(1) Résidus standardisés","="*30)
print(lb_arch)

lb_garch = acorr_ljungbox(std_resid_garch, lags=[10,20], return_df=True)
print("\n","="*30,"Ljung-Box AR(1)-GARCH(1,1) Résidus standardisés","="*30)
print(lb_garch)



# %% Deuxième méthode

# ----------------- Recherche modèle de moyenne -----------------

p_max = 5
q_max = 5
results_mean = []

for p in range(p_max+1):
    for q in range(p_max+1):
        try:
            # Estimation ARMA(p,q) (ARIMA avec d=0)
            model = sm.tsa.ARIMA(close_dif, order=(p,0,q))
            fit = model.fit()

            # Résidus
            resid = fit.resid

            # Test de Ljung-Box (lags 10 et 20)
            lb = acorr_ljungbox(resid, lags=[10,20], return_df=True)

            results_mean.append({
                "p": p, "q": q,
                "AIC": fit.aic, "BIC": fit.bic,
                "LB10_pvalue": lb["lb_pvalue"].iloc[0],
                "LB20_pvalue": lb["lb_pvalue"].iloc[1]
            })
        except Exception as e:
            # Certains modèles peuvent échouer (non inversibles)
            results_mean.append({
                "p": p, "q": q,
                "AIC": np.nan, "BIC": np.nan,
                "LB10_pvalue": np.nan, "LB20_pvalue": np.nan
            })

# Résultats sous forme de tableau
df_results_mean = pd.DataFrame(results_mean)

# Classement : d'abord par BIC croissant, puis par AIC
df_sorted_mean = df_results_mean.sort_values(by=["BIC","AIC"])
print(df_sorted_mean.head(50))

# ----------------- Modèle de moyenne MA(1) -----------------

ma1 = sm.tsa.ARIMA(close_dif, order=(0,0,1)).fit()
resid_ma1 = ma1.resid

arch_ma1 = het_arch(resid_ma1, nlags=10)
print("\n","="*30,"ARCH LM AM(1)", "="*30)
print("LM stat:", arch_ma1[0], "p-value:", arch_ma1[1])

# ----------------- Recherche modèle volatilité -----------------

vol_models = [
    ("ARCH", 1, 0, "ARCH(1)"),
    ("GARCH", 1, 1, "GARCH(1,1)"),
    ("EGARCH", 1, 1, "EGARCH(1,1)")
]

dists = ["t", "skewt"]

results = []

for vol, p, q, name in vol_models:
    for dist in dists:
        try:
            model = arch_model(resid_ma1*100, mean="Zero", vol=vol, p=p, q=q, dist=dist)
            fit = model.fit(disp="off")

            # Résidus standardisés
            std_resid = fit.std_resid.dropna()

            # Ljung-Box sur résidus et résidus^2
            lb_resid = acorr_ljungbox(std_resid, lags=[10,20], return_df=True)
            lb_sqresid = acorr_ljungbox(std_resid**2, lags=[10,20], return_df=True)

            pass_lb = (lb_resid["lb_pvalue"].min() > 0.05) and (lb_sqresid["lb_pvalue"].min() > 0.05)

            results.append({
                "Modèle": name,
                "Dist": dist,
                "AIC": fit.aic,
                "BIC": fit.bic,
                "LB_resid_p10": lb_resid["lb_pvalue"].iloc[0],
                "LB_resid_p20": lb_resid["lb_pvalue"].iloc[1],
                "LB_sqresid_p10": lb_sqresid["lb_pvalue"].iloc[0],
                "LB_sqresid_p20": lb_sqresid["lb_pvalue"].iloc[1],
                "Valide": pass_lb
            })
        except Exception as e:
            results.append({
                "Modèle": name,
                "Dist": dist,
                "AIC": None, "BIC": None,
                "LB_resid_p10": None, "LB_resid_p20": None,
                "LB_sqresid_p10": None, "LB_sqresid_p20": None,
                "Valide": False
            })

df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values(by="BIC")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("\n",df_sorted)

# ----------------- MA(1) - GARCH(1,1) -----------------

model = arch_model(resid_ma1*100, mean="Zero", vol="GARCH", p=1, q=1, dist="skewt")
fit = model.fit(disp="off")

std_resid = fit.std_resid.dropna()

print("\n", "Test Ljung Box sur les résidus")
print(acorr_ljungbox(std_resid, lags=[10,20], return_df=True))
print("\n", "Test Ljung Box sur les résidus au carré")
print(acorr_ljungbox(std_resid**2, lags=[10,20], return_df=True))

plt.hist(std_resid, bins=40, density=True)
plt.title("Résidus standardisés MA(1)-GARCH(1,1) skew-t")
plt.show()

sm.qqplot(std_resid, line="s")
plt.title("QQ-plot des résidus standardisés (vs normale)")
plt.show()

acf_fit = acf(std_resid, nlags=30)
pacf_fit = pacf(std_resid, nlags=30, method="ywm")
lags_fit = np.arange(len(acf_fit))
conf = 1.96/np.sqrt(len(close_dif))

plt.figure(figsize=(10,4))
plt.bar(lags_fit, acf_fit, width=0.15, color="steelblue", edgecolor="steelblue")  # width plus petit = barres plus fines
plt.axhline(y=0, color="black", linewidth=0.8)
plt.axhline(y=conf, color="red", linestyle="--", linewidth=0.7)
plt.axhline(y=-conf, color="red", linestyle="--", linewidth=0.7)
plt.title("ACF (barres fines)", fontsize=14)
plt.show()

plt.figure(figsize=(10,4))
plt.bar(lags_fit, pacf_fit, width=0.15, color="steelblue", edgecolor="steelblue")
plt.axhline(y=0, color="black", linewidth=0.8)
plt.axhline(y=conf, color="red", linestyle="--", linewidth=0.7)
plt.axhline(y=-conf, color="red", linestyle="--", linewidth=0.7)
plt.title("PACF (barres fines)", fontsize=14)
plt.show()

cond_vol = fit.conditional_volatility
cond_vol = cond_vol.reindex(close_dif.index)
rolling_vol = close_dif.rolling(window=20).std() * np.sqrt(252)
cond_vol_ann = (cond_vol/100) * np.sqrt(252)

plt.figure(figsize=(12,6))
plt.plot(cond_vol_ann, label="Volatilité conditionnelle (MA(1)-GARCH(1,1) skew-t)", color="red")
plt.plot(rolling_vol, label="Volatilité empirique (écart-type glissant 20j)", color="blue", alpha=0.7)
plt.legend()
plt.title("Comparaison volatilité conditionnelle vs volatilité empirique")
plt.show()

# ----------------- MA(1) - EGARCH(1) -----------------

model_EGARCH = arch_model(resid_ma1*100, mean="Zero", vol="EGARCH", p=1, q=1, dist="skewt")
fit_EGARCH = model_EGARCH.fit(disp="off")

std_resid_EGARCH = fit_EGARCH.std_resid.dropna()

print("\n", "Test Ljung Box sur les résidus (EGARCH)")
print(acorr_ljungbox(std_resid_EGARCH, lags=[10,20], return_df=True))
print("\n", "Test Ljung Box sur les résidus au carré (EGARCH)")
print(acorr_ljungbox(std_resid_EGARCH**2, lags=[10,20], return_df=True))

plt.hist(std_resid_EGARCH, bins=40, density=True)
plt.title("Résidus standardisés MA(1)-EGARCH(1,1)skew-t")
plt.show()

sm.qqplot(std_resid_EGARCH, line="s")
plt.title("QQ-plot des résidus standardisés (vs normale) (EGARCH)")
plt.show()

acf_fit_EGARCH = acf(std_resid_EGARCH, nlags=30)
pacf_fit_EGARCH = pacf(std_resid_EGARCH, nlags=30, method="ywm")
lags_fit_EGARCH = np.arange(len(acf_fit_EGARCH))

plt.figure(figsize=(10,4))
plt.bar(lags_fit_EGARCH, acf_fit_EGARCH, width=0.15, color="steelblue", edgecolor="steelblue")  # width plus petit = barres plus fines
plt.axhline(y=0, color="black", linewidth=0.8)
plt.axhline(y=conf, color="red", linestyle="--", linewidth=0.7)
plt.axhline(y=-conf, color="red", linestyle="--", linewidth=0.7)
plt.title("ACF (barres fines) (EGARCH)", fontsize=14)
plt.show()

plt.figure(figsize=(10,4))
plt.bar(lags_fit_EGARCH, pacf_fit_EGARCH, width=0.15, color="steelblue", edgecolor="steelblue")
plt.axhline(y=0, color="black", linewidth=0.8)
plt.axhline(y=conf, color="red", linestyle="--", linewidth=0.7)
plt.axhline(y=-conf, color="red", linestyle="--", linewidth=0.7)
plt.title("PACF (barres fines) (EGARCH)", fontsize=14)
plt.show()

cond_vol_EGARCH = fit_EGARCH.conditional_volatility
cond_vol_EGARCH = cond_vol_EGARCH.reindex(close_dif.index)
cond_vol_ann_EGARCH = (cond_vol_EGARCH/100) * np.sqrt(252)

plt.figure(figsize=(12,6))
plt.plot(cond_vol_ann_EGARCH, label="Volatilité conditionnelle (MA(1)-EGARCH(1,1) skew-t)", color="red")
plt.plot(rolling_vol, label="Volatilité empirique (écart-type glissant 20j)", color="blue", alpha=0.7)
plt.legend()
plt.title("Comparaison volatilité conditionnelle vs volatilité empirique (EGARCH)")
plt.show()

#%%

# ----------------- ARCH(1) PURE -----------------

# vol="ARCH" avec p=1 (modèle de Engle, 1982)
# On garde mean="Constant" et dist="skewt" pour la cohérence
model_ARCH = arch_model(close_dif_scaled, mean="Constant", vol="ARCH", p=1, dist="skewt")
fit_ARCH = model_ARCH.fit(disp="off")

# Extraction des résidus standardisés
std_resid_ARCH = fit_ARCH.std_resid.dropna()

print("\n" + "="*30)
print("DIAGNOSTICS ARCH(1) PURE")
print("="*30)

lb_resid_arch = acorr_ljungbox(std_resid_ARCH, lags=[10,20], return_df=True)
lb_sq_arch = acorr_ljungbox(std_resid_ARCH**2, lags=[10,20], return_df=True)

print("\nLjung-Box sur résidus (Moyenne) :\n", lb_resid_arch)
print("\nLjung-Box sur résidus² (Variance) :\n", lb_sq_arch)

lm_test_ARCH = het_arch(std_resid_ARCH)
print(f"\nTest ARCH-LM (ARCH) :")
print(f"Statistique LM : {lm_test_ARCH[0]:.4f}, p-value : {lm_test_ARCH[1]:.4f}")
# --- Graphiques de Diagnostic ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Histogramme
ax[0].hist(std_resid_ARCH, bins=40, density=True, color='lightgray', edgecolor='black', alpha=0.7)
ax[0].set_title("Distribution des résidus (ARCH)")

# QQ-Plot
sm.qqplot(std_resid_ARCH, line="s", ax=ax[1])
ax[1].set_title("QQ-Plot (ARCH vs Normale)")
plt.show()

# --- Visualisation de la Volatilité ---
cond_vol_ann_ARCH = (fit_ARCH.conditional_volatility / 100) * np.sqrt(252)
cond_vol_ann_ARCH = cond_vol_ann_ARCH.reindex(close_dif.index)

plt.figure(figsize=(12, 6))
plt.plot(cond_vol_ann_ARCH, label="Volatilité ARCH(1)", color="black", lw=1.2)
plt.plot(rolling_vol, label="Volatilité Historique (20j)", color="orange", alpha=0.4, ls="--")
plt.title("Volatilité Conditionnelle ARCH(1)", fontsize=14)
plt.ylabel("Volatilité Annualisée")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

#%%
# ----------------- GARCH(1,1) PURE (Sans Moyenne) -----------------

# On utilise mean="Zero" car on modélise directement la variance des résidus/rendements
model_GARCH_pure = arch_model(close_dif_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt")
fit_GARCH_pure = model_GARCH_pure.fit(disp="off")

# Extraction des résidus standardisés
std_resid_GARCH = fit_GARCH_pure.std_resid.dropna()

# --- Diagnostics ---
print("\n" + "="*30)
print("DIAGNOSTICS GARCH(1,1) PURE")
print("="*30)

lb_resid = acorr_ljungbox(std_resid_GARCH, lags=[10,20], return_df=True)
lb_squared = acorr_ljungbox(std_resid_GARCH**2, lags=[10,20], return_df=True)

print("\nLjung-Box sur résidus (Bruit Blanc ?) :\n", lb_resid)
print("\nLjung-Box sur résidus² (Effet ARCH éliminé ?) :\n", lb_squared)

lm_test = het_arch(std_resid_GARCH)
print(f"\nTest ARCH-LM (GARCH) :")
print(f"Statistique LM : {lm_test[0]:.4f}, p-value : {lm_test[1]:.4f}")

# --- Graphiques de Diagnostic ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Histogramme
ax[0].hist(std_resid_GARCH, bins=40, density=True, color='skyblue', edgecolor='black', alpha=0.7)
ax[0].set_title("Distribution des résidus (GARCH)")

# QQ-Plot
sm.qqplot(std_resid_GARCH, line="s", ax=ax[1])
ax[1].set_title("QQ-Plot (GARCH vs Normale)")
plt.show()

# --- Visualisation de la Volatilité ---
# On récupère la volatilité conditionnelle
cond_vol_GARCH = fit_GARCH_pure.conditional_volatility

# Annualisation (retour à l'échelle réelle : /100 car on a multiplié les données par 100)
# Rappel : volatilité_journalière * sqrt(252) = volatilité_annuelle
cond_vol_ann_GARCH = (cond_vol_GARCH / 100) * np.sqrt(252)
cond_vol_ann_GARCH = cond_vol_ann_GARCH.reindex(close_dif.index)

plt.figure(figsize=(12, 6))
plt.plot(cond_vol_ann_GARCH, label="Volatilité GARCH(1,1) Pure", color="teal", lw=1.5)
plt.plot(rolling_vol, label="Volatilité Historique (20j)", color="orange", alpha=0.6, linestyle="--")
plt.title("Évolution de la Volatilité Conditionnelle GARCH(1,1)", fontsize=14)
plt.ylabel("Volatilité Annualisée")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%

# ----------------- EGARCH(1,1,1) PURE -----------------

# vol="EGARCH" avec p=1, o=1, q=1
model_EGARCH = arch_model(close_dif_scaled, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="skewt")
fit_EGARCH = model_EGARCH.fit(disp="off")

std_resid_EG = fit_EGARCH.std_resid.dropna()

print("\n" + "="*30)
print("DIAGNOSTICS EGARCH(1,1,1)")
print("="*30)

lb_resid_eg = acorr_ljungbox(std_resid_EG, lags=[10,20], return_df=True)
lb_sq_eg = acorr_ljungbox(std_resid_EG**2, lags=[10,20], return_df=True)

print("\nLjung-Box sur résidus :\n", lb_resid_eg)
print("\nLjung-Box sur résidus² :\n", lb_sq_eg)

lm_test_eg = het_arch(std_resid_eg)
print(f"\nTest ARCH-LM (EGARCH) :")
print(f"Statistique LM : {lm_test_eg[0]:.4f}, p-value : {lm_test_eg[1]:.4f}")
# --- Graphiques ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].hist(std_resid_EG, bins=40, density=True, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax[0].set_title("Distribution des résidus (EGARCH)")
sm.qqplot(std_resid_EG, line="s", ax=ax[1])
ax[1].set_title("QQ-Plot (EGARCH)")
plt.show()

# --- Volatilité Annualisée ---
cond_vol_ann_EG = (fit_EGARCH.conditional_volatility / 100) * np.sqrt(252)
cond_vol_ann_EG = cond_vol_ann_EG.reindex(close_dif.index)

plt.figure(figsize=(12, 6))
plt.plot(cond_vol_ann_EG, label="Volatilité EGARCH", color="darkgreen", lw=1.5)
plt.plot(rolling_vol, label="Volatilité Historique (20j)", color="orange", alpha=0.4, ls="--")
plt.title("Volatilité Conditionnelle EGARCH")
plt.legend()
plt.show()

# %%

# ----------------- GJR-GARCH(1,1,1) PURE -----------------

# o=1 active l'asymétrie (GJR-GARCH)
model_GJR = arch_model(close_dif_scaled, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewt")
fit_GJR = model_GJR.fit(disp="off")

std_resid_GJR = fit_GJR.std_resid.dropna()

print("\n" + "="*30)
print("DIAGNOSTICS GJR-GARCH(1,1,1)")
print("="*30)

lb_resid_gjr = acorr_ljungbox(std_resid_GJR, lags=[10,20], return_df=True)
lb_sq_gjr = acorr_ljungbox(std_resid_GJR**2, lags=[10,20], return_df=True)

print("\nLjung-Box sur résidus :\n", lb_resid_gjr)
print("\nLjung-Box sur résidus² :\n", lb_sq_gjr)

lm_test_gjr = het_arch(std_resid_gjr)
print(f"\nTest ARCH-LM (GARCH) :")
print(f"Statistique LM : {lm_test_gjr[0]:.4f}, p-value : {lm_test_gjr[1]:.4f}")

# --- Graphiques ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].hist(std_resid_GJR, bins=40, density=True, color='salmon', edgecolor='black', alpha=0.7)
ax[0].set_title("Distribution des résidus (GJR-GARCH)")
sm.qqplot(std_resid_GJR, line="s", ax=ax[1])
ax[1].set_title("QQ-Plot (GJR-GARCH)")
plt.show()

# --- Volatilité Annualisée ---
cond_vol_ann_GJR = (fit_GJR.conditional_volatility / 100) * np.sqrt(252)
cond_vol_ann_GJR = cond_vol_ann_GJR.reindex(close_dif.index)

plt.figure(figsize=(12, 6))
plt.plot(cond_vol_ann_GJR, label="Volatilité GJR-GARCH", color="darkred", lw=1.5)
plt.plot(rolling_vol, label="Volatilité Historique (20j)", color="orange", alpha=0.4, ls="--")
plt.title("Volatilité Conditionnelle GJR-GARCH (Asymétrique)")
plt.legend()
plt.show()

# %%

models_fits = {
    "ARCH(1)": fit_ARCH,
    "GARCH(1,1)": fit_GARCH_pure,
    "GJR-GARCH(1,1,1)": fit_GJR,
    "EGARCH(1,1,1)": fit_EGARCH
}

summary_data = []

for name, fit in models_fits.items():
    # Extraction des résidus standardisés
    std_res = fit.std_resid.dropna()
    
    # 1. Ljung-Box (Lag 10)
    lb_res = acorr_ljungbox(std_res, lags=[10], return_df=True)
    lb_sqres = acorr_ljungbox(std_res**2, lags=[10], return_df=True)
    
    # 2. Test ARCH-LM (Nouveau)
    # Retourne (lm_stat, p_value, f_stat, f_p_value)
    arch_lm_p = het_arch(std_res)[1]
    
    # Stockage des résultats
    summary_data.append({
        "Modèle": name,
        "AIC": round(fit.aic, 2),
        "BIC": round(fit.bic, 2),
        "LB Rés² (p-val)": round(lb_sqres["lb_pvalue"].iloc[0], 4),
        "ARCH-LM (p-val)": round(arch_lm_p, 4) # <--- La preuve ultime
    })

# Création du DataFrame final
df_comparatif = pd.DataFrame(summary_data)

# Affichage avec style
print("\n" + "="*80)
print("TABLEAU COMPARATIF DES MODÈLES DE VOLATILITÉ")
print("="*80)
print(df_comparatif.to_string(index=False))
print("="*80)

# %% Prévision

# ----------------- Fenêtre d'entrainement -----------------

train_start = "2000-01-01"
train_end   = "2009-12-31"
train = close_dif.loc[train_start:train_end]
train_df = train.reset_index()
train_df.columns = ["Date", "Close Diff"]
print("\n",train_df.head(),"\n")
# ----------------- Modèle MA(1) sur la moyenne -----------------

ma1_prev = sm.tsa.ARIMA(train, order=(0,0,1)).fit()
resid_ma1_prev = ma1_prev.resid

# ----------------- GARCH(1,1) skew-t sur les résidus -----------------

am_prev = arch_model(resid_ma1_prev*100, mean="Zero", vol="GARCH", p=1, q=1, dist="skewt")
fit_prev = am_prev.fit(disp="off")

# ----------------- Première prévision -----------------

first_oos_date = close_dif.loc["2010-01-01":].index[0]

first = fit_prev.forecast(horizon=1, reindex=False)
first_pred_var_next = first.variance.values[-1, 0] / (100**2)
first_pred_sigma_next = np.sqrt(first_pred_var_next)
first_pred_sigma_next_ann = first_pred_sigma_next * np.sqrt(252)

print("\n",f"Première prévision OOS pour {first_oos_date.date()}:")
print("\n",f"Variance prévue (journalière, décimal): {first_pred_var_next:.6e}")
print("\n",f"Volatilité prévue (σ, journalière): {first_pred_sigma_next:.6f}")
print("\n",f"Volatilité prévue annualisée: {first_pred_sigma_next_ann:.2%}")

first_realized_var_next = close_dif.loc[first_oos_date]**2
print("\n",f"Variance réalisée (proxy r^2) pour {first_oos_date.date()}: {first_realized_var_next:.6e}")

first_qlike = np.log(first_pred_var_next) + first_realized_var_next / first_pred_var_next
first_mse = (first_realized_var_next - first_pred_var_next)**2
first_mae = abs(abs(close_dif.loc[first_oos_date]) - np.sqrt(first_pred_var_next))

print("\n",f"Date OOS : {first_oos_date.date()}")
print("\n",f"Variance prédite   : {first_pred_var_next:.6e}")
print("\n",f"Variance réalisée  : {first_realized_var_next:.6e}")
print("\n",f"QLIKE : {first_qlike:.6f}")
print("\n",f"MSE   : {first_mse:.6e}")
print("\n",f"MAE   : {first_mae:.6f}")

# ----------------- Test OSS GARCH-----------------


start_test = "2010-01-01"
end_test   = "2024-12-31"
test = close_dif.loc[start_test:end_test]
test_df = test.reset_index()
test_df.columns = ["Date", "Close Diff"]
print("\n",test_df.head(),"\n")

refit = 20
train_years = 10
alpha = 0.01 #VaR à 1%

pred_var, pred_sigma, pred_sigma_ann = [], [], []
real_var, returns_list, var_skewt = [], [], []

fit_oss = None
q_skewt = None
for i, date in tqdm(enumerate(test_df["Date"]), total=len(test_df)):
    
    if i % refit == 0 or fit_oss is None:
        
        train_start = date - pd.DateOffset(years=train_years)
        train = close_dif.loc[train_start:date - pd.Timedelta(days=1)]
        
        ma1 = sm.tsa.ARIMA(train, order=(0,0,1)).fit()
        resid = ma1.resid
        
        am = arch_model(resid*100, mean="Zero", vol="GARCH", p=1, q=1, dist="skewt")
        fit_oss = am.fit(disp="off")
        
        nu = float(fit_oss.params.get("eta", fit_oss.params.get("nu")))
        lam = float(fit_oss.params["lambda"])
        
        q_skewt = skewt_quantile_arch(alpha, lam, nu)
        
    f = fit_oss.forecast(horizon=1, reindex=False)
    var_pred = f.variance.values[-1,0] / (100**2)
    sigma_pred = np.sqrt(var_pred)
    sigma_pred_ann = sigma_pred * np.sqrt(252) 
    
    var_real = close_dif.loc[date]**2
    r_t = close_dif.loc[date]
    
    var_t = q_skewt * sigma_pred
    
    pred_var.append(var_pred)
    pred_sigma.append(sigma_pred)
    pred_sigma_ann.append(sigma_pred_ann)
    real_var.append(var_real)
    returns_list.append(r_t)
    var_skewt.append(var_t)
    
oos = pd.DataFrame({
    "Pred_Var": pred_var,
    "Pred_Sigma": pred_sigma,
    "Pred_Sigma_Ann": pred_sigma_ann,
    "Real_Var": real_var,
    "Return": returns_list,
    "VaR_1pct": var_skewt
}, index=test_df["Date"][:len(pred_sigma)])
oos["Real_Sigma"] = np.sqrt(oos["Real_Var"]) 
oos["Real_Sigma_Ann"] = oos["Real_Sigma"] * np.sqrt(252)
print(oos.head())
        
plt.figure(figsize=(12,6))
plt.plot(oos.index, oos["Pred_Sigma"], label="Volatilité prédite (σ_t)", color="red")
plt.plot(oos.index, oos["Real_Sigma"], label="Volatilité réalisée (|r_t|)", color="blue", alpha=0.7)
plt.title("Comparaison volatilité prédite (GARCH) vs réalisée (journalière)")
plt.legend() 
plt.show()

plt.figure(figsize=(12,6))
plt.plot(oos.index, oos["Pred_Sigma_Ann"], label="Volatilité prédite annualisée", color="red")
plt.plot(oos.index, oos["Real_Sigma"]*np.sqrt(252), label="Volatilité réalisée annualisée", color="blue", alpha=0.7)
plt.title("Comparaison volatilité annualisée : prédite (GARCH) vs réalisée")
plt.legend()
plt.show()

oos["QLIKE"] = np.log(oos["Pred_Var"]) + oos["Real_Var"] / oos["Pred_Var"]
oos["MSE"] = (oos["Real_Var"] - oos["Pred_Var"])**2
oos["MAE"] = (oos["Real_Sigma"] - oos["Pred_Sigma"]).abs()

metrics_summary = oos[["QLIKE","MSE","MAE"]].mean()
print("\n","Métriques moyennes OOS (2010–2024) (GARCH) :","\n")
print(metrics_summary)

exceptions = (oos["Return"] < oos["VaR_1pct"]).sum()
print(f"Nombre d'exceptions VaR 1% (skew-t dynamique) (GARCH) : {exceptions} sur {len(oos)} jours")

res_pof = kupiec_pof(oos["Return"], oos["VaR_1pct"], alpha=0.01)

print("\n=== Test de Kupiec (POF) VaR 1% (GARCH) ===")
for k, v in res_pof.items():
    if isinstance(v, float):
        print(f"{k:15s}: {v:.4f}")
    else:
        print(f"{k:15s}: {v}")

res_ind = christoffersen_ind(oos["Return"], oos["VaR_1pct"])

print("\n=== Test Christoffersen IND (Indépendance) (GARCH) ===")
for k, v in res_ind.items():
    if isinstance(v, float):
        print(f"{k:10s}: {v:.4f}")
    else:
        print(f"{k:10s}: {v}")

oos.to_csv(
    "C:\\Users\\hugom\\OneDrive - Aescra Emlyon Business School\\Mines de Saint-Etienne\\3A\\Projet Indus\\Robo advisor\\Sorties\\oos_garch_sp500.csv",
    index_label="Date"
)

# ----------------- Test OSS EGARCH-----------------

pred_var_EGARCH, pred_sigma_EGARCH, pred_sigma_ann_EGARCH, var_skewt_EGARCH  = [], [], [], []

fit_oss_EGARCH = None
q_skewt_EGARCH = None
for i, date in tqdm(enumerate(test_df["Date"]), total=len(test_df)):
    
    if i % refit == 0 or fit_oss_EGARCH is None:
        
        train_start = date - pd.DateOffset(years=train_years)
        train = close_dif.loc[train_start:date - pd.Timedelta(days=1)]
        
        ma1 = sm.tsa.ARIMA(train, order=(0,0,1)).fit()
        resid = ma1.resid
        
        am = arch_model(resid*100, mean="Zero", vol="EGARCH", p=1, q=1, dist="skewt")
        fit_oss_EGARCH = am.fit(disp="off")
        
        nu = float(fit_oss_EGARCH.params.get("eta", fit_oss_EGARCH.params.get("nu")))
        lam = float(fit_oss_EGARCH.params["lambda"])
        
        q_skewt_EGARCH = skewt_quantile_arch(alpha, lam, nu)
        
    f = fit_oss_EGARCH.forecast(horizon=1, reindex=False)
    var_pred = f.variance.values[-1,0] / (100**2)
    sigma_pred = np.sqrt(var_pred)
    sigma_pred_ann = sigma_pred * np.sqrt(252) 
    
    var_t = q_skewt_EGARCH * sigma_pred
    
    pred_var_EGARCH.append(var_pred)
    pred_sigma_EGARCH.append(sigma_pred)
    pred_sigma_ann_EGARCH.append(sigma_pred_ann)
    var_skewt_EGARCH.append(var_t)
    
oos_EGARCH = pd.DataFrame({
    "Pred_Var": pred_var_EGARCH,
    "Pred_Sigma": pred_sigma_EGARCH,
    "Pred_Sigma_Ann": pred_sigma_ann_EGARCH,
    "Real_Var": real_var,
    "Return": returns_list,
    "VaR_1pct": var_skewt_EGARCH
}, index=test_df["Date"][:len(pred_sigma_EGARCH)])
oos_EGARCH["Real_Sigma"] = np.sqrt(oos_EGARCH["Real_Var"]) 
oos_EGARCH["Real_Sigma_Ann"] = oos_EGARCH["Real_Sigma"] * np.sqrt(252)
print(oos_EGARCH.head())
        
plt.figure(figsize=(12,6))
plt.plot(oos_EGARCH.index, oos_EGARCH["Pred_Sigma"], label="Volatilité prédite (σ_t)", color="red")
plt.plot(oos_EGARCH.index, oos_EGARCH["Real_Sigma"], label="Volatilité réalisée (|r_t|)", color="blue", alpha=0.7)
plt.title("Comparaison volatilité prédite (EGARCH) vs réalisée (journalière)")
plt.legend()
plt.show() 

plt.figure(figsize=(12,6))
plt.plot(oos_EGARCH.index, oos_EGARCH["Pred_Sigma_Ann"], label="Volatilité prédite annualisée", color="red")
plt.plot(oos_EGARCH.index, oos_EGARCH["Real_Sigma"]*np.sqrt(252), label="Volatilité réalisée annualisée", color="blue", alpha=0.7)
plt.title("Comparaison volatilité annualisée : prédite (EGARCH) vs réalisée")
plt.legend()
plt.show()

oos_EGARCH["QLIKE"] = np.log(oos_EGARCH["Pred_Var"]) + oos_EGARCH["Real_Var"] / oos_EGARCH["Pred_Var"]
oos_EGARCH["MSE"] = (oos_EGARCH["Real_Var"] - oos_EGARCH["Pred_Var"])**2
oos_EGARCH["MAE"] = (oos_EGARCH["Real_Sigma"] - oos_EGARCH["Pred_Sigma"]).abs()

metrics_summary_EGARCH = oos_EGARCH[["QLIKE","MSE","MAE"]].mean()
print("\n","Métriques moyennes OOS (EGARCH) (2010–2024) :","\n")
print(metrics_summary_EGARCH)

exceptions_EGARCH = (oos_EGARCH["Return"] < oos_EGARCH["VaR_1pct"]).sum()
print(f"Nombre d'exceptions VaR 1% (skew-t dynamique) (EGARCH) : {exceptions} sur {len(oos)} jours")

res_pof_EGARCH = kupiec_pof(oos_EGARCH["Return"], oos_EGARCH["VaR_1pct"], alpha=0.01)

print("\n=== Test de Kupiec (POF) VaR 1% (EGARCH) ===")
for k, v in res_pof_EGARCH.items():
    if isinstance(v, float):
        print(f"{k:15s}: {v:.4f}")
    else:
        print(f"{k:15s}: {v}")

res_ind_EGARCH = christoffersen_ind(oos_EGARCH["Return"], oos_EGARCH["VaR_1pct"])

print("\n=== Test Christoffersen IND (Indépendance) (EGARCH) ===")
for k, v in res_ind_EGARCH.items():
    if isinstance(v, float):
        print(f"{k:10s}: {v:.4f}")
    else:
        print(f"{k:10s}: {v}")


# ----------------- Test OSS GJR-GARCH-----------------

pred_var_GJRGARCH, pred_sigma_GJRGARCH, pred_sigma_ann_GJRGARCH, var_skewt_GJRGARCH  = [], [], [], []

fit_oss_GJRGARCH = None
q_skewt_GJRGARCH = None
for i, date in tqdm(enumerate(test_df["Date"]), total=len(test_df)):
    
    if i % refit == 0 or fit_oss_GJRGARCH is None:
        
        train_start = date - pd.DateOffset(years=train_years)
        train = close_dif.loc[train_start:date - pd.Timedelta(days=1)]
        
        ma1 = sm.tsa.ARIMA(train, order=(0,0,1)).fit()
        resid = ma1.resid
        
        am = arch_model(resid*100, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="skewt")
        fit_oss_GJRGARCH = am.fit(disp="off")
        
        nu = float(fit_oss_GJRGARCH.params.get("eta", fit_oss_GJRGARCH.params.get("nu")))
        lam = float(fit_oss_GJRGARCH.params["lambda"])
        
        q_skewt_GJRGARCH = skewt_quantile_arch(alpha, lam, nu)
        
    f = fit_oss_GJRGARCH.forecast(horizon=1, reindex=False)
    var_pred = f.variance.values[-1,0] / (100**2)
    sigma_pred = np.sqrt(var_pred)
    sigma_pred_ann = sigma_pred * np.sqrt(252) 
    
    var_t = q_skewt_GJRGARCH * sigma_pred
    
    pred_var_GJRGARCH.append(var_pred)
    pred_sigma_GJRGARCH.append(sigma_pred)
    pred_sigma_ann_GJRGARCH.append(sigma_pred_ann)
    var_skewt_GJRGARCH.append(var_t)
    
oos_GJRGARCH = pd.DataFrame({
    "Pred_Var": pred_var_GJRGARCH,
    "Pred_Sigma": pred_sigma_GJRGARCH,
    "Pred_Sigma_Ann": pred_sigma_ann_GJRGARCH,
    "Real_Var": real_var,
    "Return": returns_list,
    "VaR_1pct": var_skewt_GJRGARCH
}, index=test_df["Date"][:len(pred_sigma_GJRGARCH)])
oos_GJRGARCH["Real_Sigma"] = np.sqrt(oos_GJRGARCH["Real_Var"]) 
oos_GJRGARCH["Real_Sigma_Ann"] = oos_GJRGARCH["Real_Sigma"] * np.sqrt(252)
print(oos_GJRGARCH.head())
        
plt.figure(figsize=(12,6))
plt.plot(oos_GJRGARCH.index, oos_GJRGARCH["Pred_Sigma"], label="Volatilité prédite (σ_t)", color="red")
plt.plot(oos_GJRGARCH.index, oos_GJRGARCH["Real_Sigma"], label="Volatilité réalisée (|r_t|)", color="blue", alpha=0.7)
plt.title("Comparaison volatilité prédite (GJR-GARCH) vs réalisée (journalière)")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(oos_GJRGARCH.index, oos_GJRGARCH["Pred_Sigma_Ann"], label="Volatilité prédite annualisée", color="red")
plt.plot(oos_GJRGARCH.index, oos_GJRGARCH["Real_Sigma"]*np.sqrt(252), label="Volatilité réalisée annualisée", color="blue", alpha=0.7)
plt.title("Comparaison volatilité annualisée : prédite (GJR-GARCH) vs réalisée")
plt.legend()
plt.show()

oos_GJRGARCH["QLIKE"] = np.log(oos_GJRGARCH["Pred_Var"]) + oos_GJRGARCH["Real_Var"] / oos_GJRGARCH["Pred_Var"]
oos_GJRGARCH["MSE"] = (oos_GJRGARCH["Real_Var"] - oos_GJRGARCH["Pred_Var"])**2
oos_GJRGARCH["MAE"] = (oos_GJRGARCH["Real_Sigma"] - oos_GJRGARCH["Pred_Sigma"]).abs()

metrics_summary_GJRGARCH = oos_GJRGARCH[["QLIKE","MSE","MAE"]].mean()
print("\n","Métriques moyennes OOS (GJR-GARCH) (2010–2024) :","\n")
print(metrics_summary_GJRGARCH)

exceptions_GJRGARCH = (oos_GJRGARCH["Return"] < oos_GJRGARCH["VaR_1pct"]).sum()
print(f"Nombre d'exceptions VaR 1% (skew-t dynamique) (GJR-GARCH) : {exceptions} sur {len(oos)} jours")

res_pof_GJRGARCH = kupiec_pof(oos_GJRGARCH["Return"], oos_GJRGARCH["VaR_1pct"], alpha=0.01)

print("\n=== Test de Kupiec (POF) VaR 1% (GJR-GARCH) ===")
for k, v in res_pof_GJRGARCH.items():
    if isinstance(v, float):
        print(f"{k:15s}: {v:.4f}")
    else:
        print(f"{k:15s}: {v}")

res_ind_GJRGARCH = christoffersen_ind(oos_GJRGARCH["Return"], oos_GJRGARCH["VaR_1pct"])

print("\n=== Test Christoffersen IND (Indépendance) (GJR-GARCH) ===")
for k, v in res_ind_GJRGARCH.items():
    if isinstance(v, float):
        print(f"{k:10s}: {v:.4f}")
    else:
        print(f"{k:10s}: {v}")

# ----------------- Test OSS Comparaison -----------------

plt.figure(figsize=(12,6))
plt.plot(oos.index, oos["Pred_Sigma"], label="Volatilité prédite (σ_t)", color="red")
plt.plot(oos.index, oos["Real_Sigma"], label="Volatilité réalisée (|r_t|)", color="blue", alpha=0.7)
plt.title("Comparaison volatilité prédite (GARCH) vs réalisée (journalière)")
plt.legend() 
plt.show()

plt.figure(figsize=(12,6))
plt.plot(oos.index, oos["Pred_Sigma_Ann"], label="Volatilité prédite annualisée", color="red")
plt.plot(oos.index, oos["Real_Sigma"]*np.sqrt(252), label="Volatilité réalisée annualisée", color="blue", alpha=0.7)
plt.title("Comparaison volatilité annualisée : prédite (GARCH) vs réalisée")
plt.legend()
plt.show()

oos["QLIKE"] = np.log(oos["Pred_Var"]) + oos["Real_Var"] / oos["Pred_Var"]
oos["MSE"] = (oos["Real_Var"] - oos["Pred_Var"])**2
oos["MAE"] = (oos["Real_Sigma"] - oos["Pred_Sigma"]).abs()

metrics_summary = oos[["QLIKE","MSE","MAE"]].mean()
print("\n","Métriques moyennes OOS (2010–2024) (GARCH) :","\n")
print(metrics_summary)

exceptions = (oos["Return"] < oos["VaR_1pct"]).sum()
print(f"Nombre d'exceptions VaR 1% (skew-t dynamique) (GARCH) : {exceptions} sur {len(oos)} jours")

res_pof = kupiec_pof(oos["Return"], oos["VaR_1pct"], alpha=0.01)

print("\n=== Test de Kupiec (POF) VaR 1% (GARCH) ===")
for k, v in res_pof.items():
    if isinstance(v, float):
        print(f"{k:15s}: {v:.4f}")
    else:
        print(f"{k:15s}: {v}")

res_ind = christoffersen_ind(oos["Return"], oos["VaR_1pct"])

print("\n=== Test Christoffersen IND (Indépendance) (GARCH) ===")
for k, v in res_ind.items():
    if isinstance(v, float):
        print(f"{k:10s}: {v:.4f}")
    else:
        print(f"{k:10s}: {v}")

plt.figure(figsize=(12,6))
plt.plot(oos_EGARCH.index, oos_EGARCH["Pred_Sigma"], label="Volatilité prédite (σ_t)", color="red")
plt.plot(oos_EGARCH.index, oos_EGARCH["Real_Sigma"], label="Volatilité réalisée (|r_t|)", color="blue", alpha=0.7)
plt.title("Comparaison volatilité prédite (EGARCH) vs réalisée (journalière)")
plt.legend()
plt.show() 

plt.figure(figsize=(12,6))
plt.plot(oos_EGARCH.index, oos_EGARCH["Pred_Sigma_Ann"], label="Volatilité prédite annualisée", color="red")
plt.plot(oos_EGARCH.index, oos_EGARCH["Real_Sigma"]*np.sqrt(252), label="Volatilité réalisée annualisée", color="blue", alpha=0.7)
plt.title("Comparaison volatilité annualisée : prédite (EGARCH) vs réalisée")
plt.legend()
plt.show()

oos_EGARCH["QLIKE"] = np.log(oos_EGARCH["Pred_Var"]) + oos_EGARCH["Real_Var"] / oos_EGARCH["Pred_Var"]
oos_EGARCH["MSE"] = (oos_EGARCH["Real_Var"] - oos_EGARCH["Pred_Var"])**2
oos_EGARCH["MAE"] = (oos_EGARCH["Real_Sigma"] - oos_EGARCH["Pred_Sigma"]).abs()

metrics_summary_EGARCH = oos_EGARCH[["QLIKE","MSE","MAE"]].mean()
print("\n","Métriques moyennes OOS (EGARCH) (2010–2024) :","\n")
print(metrics_summary_EGARCH)

exceptions_EGARCH = (oos_EGARCH["Return"] < oos_EGARCH["VaR_1pct"]).sum()
print(f"Nombre d'exceptions VaR 1% (skew-t dynamique) (EGARCH) : {exceptions} sur {len(oos)} jours")

res_pof_EGARCH = kupiec_pof(oos_EGARCH["Return"], oos_EGARCH["VaR_1pct"], alpha=0.01)

print("\n=== Test de Kupiec (POF) VaR 1% (EGARCH) ===")
for k, v in res_pof_EGARCH.items():
    if isinstance(v, float):
        print(f"{k:15s}: {v:.4f}")
    else:
        print(f"{k:15s}: {v}")

res_ind_EGARCH = christoffersen_ind(oos_EGARCH["Return"], oos_EGARCH["VaR_1pct"])

print("\n=== Test Christoffersen IND (Indépendance) (EGARCH) ===")
for k, v in res_ind_EGARCH.items():
    if isinstance(v, float):
        print(f"{k:10s}: {v:.4f}")
    else:
        print(f"{k:10s}: {v}")

plt.figure(figsize=(12,6))
plt.plot(oos_GJRGARCH.index, oos_GJRGARCH["Pred_Sigma"], label="Volatilité prédite (σ_t)", color="red")
plt.plot(oos_GJRGARCH.index, oos_GJRGARCH["Real_Sigma"], label="Volatilité réalisée (|r_t|)", color="blue", alpha=0.7)
plt.title("Comparaison volatilité prédite (GJR-GARCH) vs réalisée (journalière)")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(oos_GJRGARCH.index, oos_GJRGARCH["Pred_Sigma_Ann"], label="Volatilité prédite annualisée", color="red")
plt.plot(oos_GJRGARCH.index, oos_GJRGARCH["Real_Sigma"]*np.sqrt(252), label="Volatilité réalisée annualisée", color="blue", alpha=0.7)
plt.title("Comparaison volatilité annualisée : prédite (GJR-GARCH) vs réalisée")
plt.legend()
plt.show()

oos_GJRGARCH["QLIKE"] = np.log(oos_GJRGARCH["Pred_Var"]) + oos_GJRGARCH["Real_Var"] / oos_GJRGARCH["Pred_Var"]
oos_GJRGARCH["MSE"] = (oos_GJRGARCH["Real_Var"] - oos_GJRGARCH["Pred_Var"])**2
oos_GJRGARCH["MAE"] = (oos_GJRGARCH["Real_Sigma"] - oos_GJRGARCH["Pred_Sigma"]).abs()

metrics_summary_GJRGARCH = oos_GJRGARCH[["QLIKE","MSE","MAE"]].mean()
print("\n","Métriques moyennes OOS (GJR-GARCH) (2010–2024) :","\n")
print(metrics_summary_GJRGARCH)

exceptions_GJRGARCH = (oos_GJRGARCH["Return"] < oos_GJRGARCH["VaR_1pct"]).sum()
print(f"Nombre d'exceptions VaR 1% (skew-t dynamique) (GJR-GARCH) : {exceptions} sur {len(oos)} jours")

res_pof_GJRGARCH = kupiec_pof(oos_GJRGARCH["Return"], oos_GJRGARCH["VaR_1pct"], alpha=0.01)

print("\n=== Test de Kupiec (POF) VaR 1% (GJR-GARCH) ===")
for k, v in res_pof_GJRGARCH.items():
    if isinstance(v, float):
        print(f"{k:15s}: {v:.4f}")
    else:
        print(f"{k:15s}: {v}")

res_ind_GJRGARCH = christoffersen_ind(oos_GJRGARCH["Return"], oos_GJRGARCH["VaR_1pct"])

print("\n=== Test Christoffersen IND (Indépendance) (GJR-GARCH) ===")
for k, v in res_ind_GJRGARCH.items():
    if isinstance(v, float):
        print(f"{k:10s}: {v:.4f}")
    else:
        print(f"{k:10s}: {v}")


































































