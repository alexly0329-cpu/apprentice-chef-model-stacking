# Apprentice Chef — Model Stacking for Revenue & Cross-Sell Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-tuning-4B32C3)
![Notebook](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Two end-to-end supervised learning studies on a meal-kit subscription business: one **regression** model that predicts customer revenue, and one **classification** model that predicts who will subscribe to a promotional campaign. Both close with a **stacked ensemble** — several base learners blended by weight, tuned explicitly to shrink the train–test gap rather than to maximise the training score.

> Coursework for DAT-5329 Machine Learning, MSc Finance, Hult International Business School — written by Alexandre Ly.

---

## Results at a glance

| Study | Target | Final model | Test score | Train–test gap |
|---|---|---|---|---|
| **Part 1 — Regression** | `log_REVENUE` | Blend: GBM 0.60 · KNN 0.20 · Lasso 0.20 | **R² 0.763** · MAPE 16.3% | 0.14 |
| **Part 2 — Classification** | `CROSS_SELL_SUCCESS` | Blend: Logistic 0.60 · GBM 0.30 · KNN 0.10 | **AUC 0.721** | 0.047 |

The headline number in both studies is the *gap*, not the peak score. An unconstrained neural network reached a 0.967 training R² and collapsed to 0.430 on test data; the blended models trade a few points of raw fit for a model that behaves the same way out of sample.

---

## What's in this repo

| File | What it does |
|---|---|
| [`A1_Part1_Revenue_Regression.ipynb`](A1_Part1_Revenue_Regression.ipynb) | EDA → feature engineering → KNN, Random Forest, GBM, Ridge, Lasso, Elastic Net, MLP → weighted stack |
| [`A1_Part2_CrossSell_Classification.ipynb`](A1_Part2_CrossSell_Classification.ipynb) | Email-domain feature engineering → Logistic, KNN, Random Forest, GBM, MLP → weighted stack |
| `data/Apprentice_Chef_Dataset.xlsx` | Source dataset (1,946 customers, 28 raw features) |
| `assets/` | Exported charts used in this README |

**View rendered** (no account needed) — [Part 1](https://nbviewer.org/github/alexly0329-cpu/apprentice-chef-model-stacking/blob/main/A1_Part1_Revenue_Regression.ipynb) · [Part 2](https://nbviewer.org/github/alexly0329-cpu/apprentice-chef-model-stacking/blob/main/A1_Part2_CrossSell_Classification.ipynb)

**Open in Colab** (to run it) — [Part 1](https://colab.research.google.com/github/alexly0329-cpu/apprentice-chef-model-stacking/blob/main/A1_Part1_Revenue_Regression.ipynb) · [Part 2](https://colab.research.google.com/github/alexly0329-cpu/apprentice-chef-model-stacking/blob/main/A1_Part2_CrossSell_Classification.ipynb)

---

## Part 1 — Predicting customer revenue

**Question.** Which behaviours drive how much a customer spends, and can we predict spend well enough to size a customer's lifetime value?

**Approach.**
- Log-transformed the target and the heavily right-skewed predictors — revenue goes from a long-tailed distribution to a near-normal one, which is what the linear members of the stack need.
- Standardised all features before the train/test split (75/25) so distance-based models (KNN) and penalised models (Ridge/Lasso) are on comparable scales.
- Tuned GBM and MLP hyperparameters with **Optuna** rather than manual grid search.
- Blended three models that see the data differently: Lasso (linear, drops weak predictors), KNN (no functional form assumption), and a depth-constrained GBM (non-linear interactions).

![Feature importance — revenue drivers](assets/feature_importance_revenue.png)

**Key findings.**
1. **`AVG_PREP_VID_TIME` is the strongest single predictor.** Time spent watching prep videos is an engagement proxy — engaged customers order more.
2. **`CONTACTS_W_CUSTOMER_SERVICE` matters more than expected**, and in the positive direction: contact volume tracks order volume, not dissatisfaction.
3. **Median meal rating peaks at 4, not 5** — a central-tendency bias in self-reported ratings that makes rating a weaker feature than it first appears.

![Blended model diagnostics](assets/blended_model_diagnostics.png)

---

## Part 2 — Predicting cross-sell subscriptions

**Question.** Who signs up for the *Halfway There* promotion, and what should marketing do with that?

**Approach.**
- Engineered a **domain group** feature by splitting the email address and bucketing domains into *professional / personal / junk* — a piece of business domain knowledge, not something a model could have found on its own.
- Stratified the train/test split to preserve the 68/32 class balance, and scored on **AUC** rather than accuracy (a naive "everyone subscribes" model already scores ~68%).
- Blended Logistic Regression, a tuned GBM, and KNN on predicted probabilities, weighted toward the most stable learner.

![Feature importance — cross-sell drivers](assets/feature_importance_crosssell.png)

**Key findings.**
1. **Email reachability is the gatekeeper.** Junk-domain customers subscribe at ~42% versus ~70% (personal) and ~80% (professional). The promotion is emailed, so deliverability *is* the conversion funnel.
2. **Cancelling before noon predicts subscription, not churn.** Pre-noon cancellation is penalty-free and signals a customer who actively manages their plan — the engaged ones, in other words.
3. **Constraining the GBM helped in Part 1 but not here.** Re-weighting toward the logistic regression (train–test gap of 0.0035) cut the blend's gap from 0.168 to 0.047 while *improving* test AUC.

---

## Running it yourself

```bash
git clone https://github.com/alexly0329-cpu/apprentice-chef-model-stacking.git
cd apprentice-chef-model-stacking
pip install -r requirements.txt
jupyter notebook
```

The notebooks resolve the dataset from `data/` automatically and fall back to downloading it from this repo, so they also run unchanged in Google Colab.

> One dependency, `baserush`, is a course-provided helper package used for `quick_tree` and `quick_neighbors`. Those cells are convenience wrappers around cross-validated tree and KNN searches; everything downstream is standard scikit-learn.

---

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `statsmodels` · `optuna` · `phik` · `seaborn` · `matplotlib`

---

## Author

**Alexandre Ly** — MSc Finance (STEM), Hult International Business School
[LinkedIn](https://linkedin.com/in/alexandre-t-ly)
