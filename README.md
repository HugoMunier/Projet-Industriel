# Classificateur de Régimes de Stress pour l'Allocation Dynamique d'Actifs

**Mines Saint-Étienne — Projet Industriel 2025–2026**

Une architecture hybride **LSTM + GARCH** pour la classification des régimes de stress de marché sur le S&P 500, couplée à une stratégie d'allocation dynamique qui pivote systématiquement entre actions (SPY), obligations d'État américaines à long terme (TLT) et or (GLD).

---

## Présentation

Plutôt que de prédire la direction du marché — un signal notoirement bruité — ce projet reformule la gestion du risque de portefeuille comme un **problème de classification de régimes**. Le modèle assigne chaque journée de bourse à l'un des trois états suivants :

| Régime | Description | Allocation cible |
|--------|-------------|-----------------|
| **Calme** | Faible volatilité, conditions normales | 100% SPY |
| **Nerveux** | Incertitude croissante, VIX élevé | 60% SPY / 40% TLT |
| **Danger** | Stress systémique, conditions de crise | 20% SPY / 80% GLD |

Le classificateur utilise une **fenêtre glissante de 60 jours** et un **décalage temporel de 2 jours** sur la variable cible, laissant au portefeuille le temps de se rééquilibrer avant que le stress ne se matérialise pleinement dans les prix.

---

## Résultats (backtest 2023 – début 2026)

| Métrique | Buy & Hold SPY | Robo GARCH | Robo GJR-GARCH |
|----------|---------------|------------|----------------|
| Rendement total (net de frais) | ~75% | **89,8%** | 80,7% |
| Volatilité annualisée | 15,2% | 12,8% | 12,5% |
| Ratio de Sharpe | 1,33 | **1,74** | 1,70 |
| Drawdown maximal | -18,8% | **-10,0%** | -10,0% |
| VaR 95% (1 jour) | -1,41% | -1,38% | **-1,34%** |

Les frais de transaction (0,1% par réallocation) sont inclus. Le nombre de réallocations sur 3 ans est de 14 (GARCH) et 20 (GJR-GARCH).

---

## Architecture

```
Variables d'entrée (quotidiennes)
  ├── Rendements logarithmiques
  ├── RSI (14 jours)
  ├── VIX (volatilité implicite)
  ├── Volatilité conditionnelle GARCH(1,1)      ← garch_feature.py
  └── Volatilité conditionnelle GJR-GARCH(1,1)  ← gjr_garch_feature.py
         │
         ▼
  BatchNormalization
         │
  LSTM (32 unités, return_sequences=True)
         │
  LSTM (16 unités)
         │
  Dense (ReLU) + Dropout (0,4)
         │
  Softmax → [P(Calme), P(Nerveux), P(Danger)]
         │
         ▼
  Buffer d'hystérésis (optimisé via Optuna)
         │
         ▼
  Décision d'allocation du portefeuille
```

Le déséquilibre de classes est géré par **class weighting** (les crises sont rares par construction). Le seuil de décision pour la classe *Danger* est fixé à 0,5 ; la classe *Nerveux* est déclenchée à 0,4.

---

## Structure du dépôt

```
├── garch_feature.py             # Feature de volatilité conditionnelle GARCH(1,1)
├── gjr_garch_feature.py         # Feature de volatilité conditionnelle GJR-GARCH(1,1)
├── conditional_volatility.py    # Pipeline complet d'analyse économétrique (famille GARCH)
├── Robo-Advisor-GARCH.ipynb     # Classificateur LSTM + backtest (moteur GARCH)
├── Robo-Advisor-GJR-GARCH.ipynb # Classificateur LSTM + backtest (moteur GJR-GARCH)
├── GOLD.ipynb                   # Analyse des corrélations : GLD vs S&P 500
├── TLT.ipynb                    # Analyse des corrélations : TLT vs S&P 500
├── AGG.ipynb                    # Analyse des corrélations : AGG vs S&P 500
└── README.md
```

---

## Installation

```bash
pip install numpy pandas arch statsmodels scikit-learn tensorflow optuna \
            pandas_market_calendars yfinance tqdm matplotlib
```

**Python 3.9+** requis.

---

## Utilisation

### Calculer les features GARCH

```python
import yfinance as yf
from garch_feature import get_garch_features
from gjr_garch_feature import get_gjr_garch_features

df = yf.download("^GSPC", start="2010-01-01")
df["GARCH_vol"]    = get_garch_features(df)
df["GJRGARCH_vol"] = get_gjr_garch_features(df)
```

### Lancer l'analyse économétrique complète

```bash
python conditional_volatility.py --data chemin/vers/SP500.csv
```

Le fichier CSV doit contenir les colonnes : `Date, Open, High, Low, Close, Volume`.

---

## Rapport

La méthodologie complète, la justification des choix de modèles et l'analyse des performances sont détaillées dans [`Projet_Industriel_Hugo_MUNIER.pdf`](./Projet_Industriel_Hugo_MUNIER.pdf).

---

## Références

- Bollerslev, T. (1986). *Generalised Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics.
- Glosten, Jagannathan & Runkle (1993). *On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks*. Journal of Finance.
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation.
- Kim & Won (2018). *Forecasting stock price index volatility: A hybrid model integrating LSTM with GARCH*. Expert Systems with Applications.
- Akiba et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD.
