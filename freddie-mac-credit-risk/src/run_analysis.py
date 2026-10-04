"""
Main entry point. Builds the loan-level dataset, compares default definitions,
trains PD models (random split and out-of-time), ranks drivers, measures realized
loss by CRT-style LTV tier, and writes aggregated CSVs to results/.

Usage (from the repo root):
    python src/run_analysis.py 2006 2007 2008 2015 2019
Set REUSE=1 to reuse the cached loan-level table in data/processed/.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
import sflld_pipeline as sp

years = sys.argv[1:] or ["2006", "2007", "2008", "2015", "2019"]
OUT = sp.OUT_DIR
cache = os.path.join(sp.CACHE_DIR, "model_data.pkl")
if os.environ.get("REUSE") and os.path.exists(cache):
    df = pd.read_pickle(cache)
else:
    df = sp.build_dataset(years)
    df.to_pickle(cache)
pd.set_option("display.width", 200)

# 1. How the definition of "default" changes the story
defs = df.groupby("vintage")[["default_naive", "default", "default_strict"]].mean()
defs.columns = ["Naive 90+ DPD (incl. forbearance)", "90+ DPD excl. forbearance (model target)",
                "Strict: ended in credit-loss event"]
print("\n=== Default rate by definition ===")
print((defs * 100).round(2))
defs.reset_index().melt(id_vars="vintage", var_name="definition", value_name="rate") \
    .to_csv(os.path.join(OUT, "tableau_default_definitions.csv"), index=False)

# 2. In-sample random split (all vintages) with the corrected target
lr, gb, res, rocs, (X_te, y_te) = sp.train_models(df)
print("\n=== Random 80/20 split, all vintages ===")
for k, v in res.items():
    print(f"{k}: AUC={v['AUC']:.3f} PR-AUC={v['PR_AUC']:.3f}")
sp.tableau_exports(df, rocs)

# 3. Out-of-time test: train on crisis-era vintages, score post-crisis vintages
train = df[df.vintage <= 2008]
print("\n=== Out-of-time: train 2006-2008, test on later vintages ===")
lr_o, gb_o, _, _, _ = sp.train_models(train, use_vintage=False)
oot_rows = []
for v in sorted(df.vintage.unique()):
    if v <= 2008:
        continue
    t = df[df.vintage == v]
    X = t[sp.NUM_FEATS + sp.CAT_FEATS]
    for name, m in [("Logistic Regression", lr_o), ("Gradient Boosting", gb_o)]:
        p = m.predict_proba(X)[:, 1]
        auc, pr = roc_auc_score(t.default, p), average_precision_score(t.default, p)
        oot_rows.append({"test_vintage": v, "model": name, "AUC": auc, "PR_AUC": pr,
                         "base_rate": t.default.mean()})
        print(f"{v} {name}: AUC={auc:.3f} PR-AUC={pr:.3f} (base rate {t.default.mean():.3%})")
pd.DataFrame(oot_rows).to_csv(os.path.join(OUT, "tableau_out_of_time.csv"), index=False)

# 4. Drivers: logistic odds ratios and gradient boosting permutation importance
names = lr.named_steps["pre"].get_feature_names_out()
coefs = lr.named_steps["clf"].coef_[0]
odds = pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
odds = odds.reindex(odds.coef.abs().sort_values(ascending=False).index)
print("\n=== Top logistic drivers (numeric features standardized: odds ratio per 1 SD) ===")
print(odds.head(12).to_string(index=False))
odds.to_csv(os.path.join(OUT, "tableau_logit_odds_ratios.csv"), index=False)

Xs = X_te.sample(20000, random_state=0) if len(X_te) > 20000 else X_te
ys = pd.Series(y_te, index=X_te.index).loc[Xs.index]
pi = permutation_importance(gb, Xs, ys, scoring="roc_auc", n_repeats=3, random_state=0)
imp = pd.DataFrame({"feature": Xs.columns, "auc_drop": pi.importances_mean}) \
        .sort_values("auc_drop", ascending=False)
print("\n=== Gradient boosting permutation importance (drop in AUC) ===")
print(imp.to_string(index=False))
imp.to_csv(os.path.join(OUT, "tableau_gb_importance.csv"), index=False)

# 5. Realized loss by CRT-style LTV tier (default rate, median severity, loss rate)
d = df.copy()
d["crt_tier"] = np.select([d.ltv.between(61, 80), d.ltv.between(81, 97)],
                          ["Low LTV 61-80 (DNA-style)", "High LTV 81-97 (HQA-style)"],
                          "Outside CRT LTV bands")
loss = d.actual_loss.where(d.severity.notna(), 0).fillna(0)
d["realized_loss"] = loss
tier = d.groupby(["vintage", "crt_tier"]).agg(
    loans=("default", "size"), pd_rate=("default", "mean"),
    lgd=("severity", "median"), realized_loss=("realized_loss", "sum"),
    orig_upb=("orig_upb", "sum")).reset_index()
tier["loss_rate_bps"] = tier.realized_loss / tier.orig_upb * 1e4
print("\n=== Realized loss by CRT-style tier ===")
print(tier.round(3).to_string(index=False))
tier.to_csv(os.path.join(OUT, "tableau_crt_tiers.csv"), index=False)
