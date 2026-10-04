"""
Portfolio-project extensions: data profile, baselines, cross-validated tuning,
final model selection, confusion matrix, and error analysis.

Usage (from the repo root, after run_analysis.py has cached the loan-level table):
    python src/model_evaluation.py
Writes aggregated CSVs and a JSON summary to results/model_evaluation/.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             fbeta_score, precision_score, recall_score, accuracy_score)
import sflld_pipeline as sp

OUT = os.path.join(sp.OUT_DIR, "model_evaluation")
os.makedirs(OUT, exist_ok=True)
SEED = 42
summary = {}
t0 = time.time()

df = pd.read_pickle(os.path.join(sp.CACHE_DIR, "model_data.pkl"))

# ---------------------------------------------------------------- 1. Data profile
num = sp.NUM_FEATS + ["cltv"]
prof = df[num].describe().T[["count", "mean", "std", "min", "50%", "max"]]
prof["missing"] = df[num].isna().sum()
prof.to_csv(os.path.join(OUT, "numeric_profile.csv"))
summary["n_loans"] = int(len(df))
summary["duplicate_loan_ids"] = int(df.loan_id.duplicated().sum())
summary["default_rate_overall"] = float(df.default.mean())
summary["default_count"] = int(df.default.sum())
summary["outcomes"] = df.outcome.value_counts().to_dict()
summary["missing"] = {k: int(v) for k, v in df[num].isna().sum().items()}
summary["first_time_buyer_not_available"] = int((df.first_time_buyer == "9").sum())
summary["msa_blank"] = int((df.msa == "").sum())
summary["corr_ltv_cltv"] = float(df[["ltv", "cltv"]].corr().iloc[0, 1])
summary["ltv_over_100"] = int((df.ltv > 100).sum())
summary["ltv_over_100_by_vintage"] = {int(k): int(v) for k, v in df[df.ltv > 100].groupby("vintage").size().items()}
summary["ltv_over_100_default_rate"] = float(df[df.ltv > 100].default.mean())
summary["share_one_unit"] = float((df.num_units == 1).mean())
summary["severity_n"] = int(df.severity.notna().sum())
summary["severity_median"] = float(df.severity.median())
summary["severity_below_0"] = int((df.severity < 0).sum())
summary["severity_above_1"] = int((df.severity > 1).sum())
summary["median_by_outcome"] = (df.groupby("default")[["fico", "ltv", "dti", "orig_rate"]]
                                .median().to_dict(orient="index"))

# Default rate by binned feature (for the exploration chart)
bins = {
    "fico": ([300, 620, 660, 700, 740, 780, 851], ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]),
    "ltv": ([0, 61, 71, 81, 91, 96, 1000], ["<=60", "61-70", "71-80", "81-90", "91-95", ">95"]),
    "dti": ([0, 20, 30, 40, 50, 66], ["<20", "20-29", "30-39", "40-49", "50-65"]),
    "orig_rate": ([0, 4, 5, 6, 7, 20], ["<4%", "4-4.99%", "5-5.99%", "6-6.99%", "7%+"]),
}
rows = []
for f, (edges, labels) in bins.items():
    b = pd.cut(df[f], edges, labels=labels, right=False)
    g = df.groupby(b, observed=True).default.agg(["size", "mean"]).reset_index()
    for _, r in g.iterrows():
        rows.append({"feature": f, "band": r[f], "loans": int(r["size"]), "default_rate": r["mean"]})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "default_rate_by_band.csv"), index=False)

# ---------------------------------------------------------------- 2. Split (same as run_analysis.py)
X = df[sp.NUM_FEATS + sp.CAT_FEATS].copy()
X["vintage"] = df.vintage.astype(str)
cats = sp.CAT_FEATS + ["vintage"]
y = df.default.values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
summary["train_n"], summary["test_n"] = int(len(X_tr)), int(len(X_te))
summary["test_default_rate"] = float(y_te.mean())

# ---------------------------------------------------------------- 3. Baselines
res = {}
p_major = np.zeros(len(y_te))
res["Baseline: predict no default"] = {
    "AUC": 0.5, "PR_AUC": float(y_te.mean()),
    "accuracy": float(accuracy_score(y_te, p_major)), "recall": 0.0}

fico_only = Pipeline([
    ("pre", ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("sc", StandardScaler())]), ["fico"])])),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
fico_only.fit(X_tr, y_tr)
p = fico_only.predict_proba(X_te)[:, 1]
res["Baseline: credit score only"] = {"AUC": float(roc_auc_score(y_te, p)),
                                      "PR_AUC": float(average_precision_score(y_te, p))}

# ---------------------------------------------------------------- 4. Cross-validated tuning (training set only)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

lr = Pipeline([
    ("pre", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), sp.NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cats)])),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
lr_search = GridSearchCV(lr, {"clf__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc", cv=cv, n_jobs=1)
lr_search.fit(X_tr, y_tr)

gb = Pipeline([
    ("pre", ColumnTransformer([
        ("num", "passthrough", sp.NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), cats)])),
    ("clf", HistGradientBoostingClassifier(max_iter=600, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=20, class_weight="balanced",
                                           random_state=SEED))])
gb_space = {
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__min_samples_leaf": [20, 100, 300],
    "clf__l2_regularization": [0.0, 1.0, 5.0],
}
gb_search = RandomizedSearchCV(gb, gb_space, n_iter=10, scoring="roc_auc", cv=cv,
                               random_state=SEED, n_jobs=1)
gb_search.fit(X_tr, y_tr)

tuning = []
for name, s in [("Logistic Regression", lr_search), ("Gradient Boosting", gb_search)]:
    cvr = pd.DataFrame(s.cv_results_)
    for _, r in cvr.iterrows():
        tuning.append({"model": name, "params": json.dumps({k.replace("clf__", ""): v for k, v in r["params"].items()}),
                       "cv_auc_mean": r["mean_test_score"], "cv_auc_std": r["std_test_score"]})
pd.DataFrame(tuning).sort_values(["model", "cv_auc_mean"], ascending=[True, False]) \
    .to_csv(os.path.join(OUT, "tuning_results.csv"), index=False)
summary["best_params"] = {
    "Logistic Regression": {k.replace("clf__", ""): v for k, v in lr_search.best_params_.items()},
    "Gradient Boosting": {k.replace("clf__", ""): v for k, v in gb_search.best_params_.items()}}
summary["best_cv_auc"] = {"Logistic Regression": float(lr_search.best_score_),
                          "Gradient Boosting": float(gb_search.best_score_)}

lr_best, gb_best = lr_search.best_estimator_, gb_search.best_estimator_
probs = {}
for name, m in [("Logistic Regression (tuned)", lr_best), ("Gradient Boosting (tuned)", gb_best)]:
    probs[name] = m.predict_proba(X_te)[:, 1]
    res[name] = {"AUC": float(roc_auc_score(y_te, probs[name])),
                 "PR_AUC": float(average_precision_score(y_te, probs[name]))}

# Bootstrap the test-set AUC difference between the two tuned models
rng = np.random.default_rng(SEED)
pl, pg = probs["Logistic Regression (tuned)"], probs["Gradient Boosting (tuned)"]
diffs = []
for _ in range(300):
    idx = rng.integers(0, len(y_te), len(y_te))
    if y_te[idx].min() == y_te[idx].max():
        continue
    diffs.append(roc_auc_score(y_te[idx], pg[idx]) - roc_auc_score(y_te[idx], pl[idx]))
summary["auc_diff_gb_minus_lr"] = {"mean": float(np.mean(diffs)),
                                   "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}

# ---------------------------------------------------------------- 5. Threshold (chosen on training data, not test)
# Hold out 25% of the training set, refit, and pick the cutoff that maximizes F2,
# which weights recall (catching defaults) twice as heavily as precision.
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=SEED)
gb_thr = clone(gb_best).fit(X_fit, y_fit)
pv = gb_thr.predict_proba(X_val)[:, 1]
grid = np.linspace(0.05, 0.95, 91)
f2 = [fbeta_score(y_val, pv >= t, beta=2) for t in grid]
thr = float(grid[int(np.argmax(f2))])
summary["threshold"] = thr

final_name = "Gradient Boosting (tuned)"
pf = probs[final_name]
pred = (pf >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
summary["confusion_matrix"] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
summary["at_threshold"] = {"accuracy": float(accuracy_score(y_te, pred)),
                           "precision": float(precision_score(y_te, pred)),
                           "recall": float(recall_score(y_te, pred)),
                           "flagged_share": float(pred.mean())}
res[final_name].update(summary["at_threshold"])
pd.DataFrame(res).T.to_csv(os.path.join(OUT, "model_comparison.csv"))

# ---------------------------------------------------------------- 6. Error analysis on the test set
te = df.loc[X_te.index].copy()
te["p"] = pf
te["pred"] = pred
te["y"] = y_te
by_v = []
for v, g in te.groupby("vintage"):
    by_v.append({"group": f"vintage {v}", "loans": len(g), "default_rate": g.y.mean(),
                 "AUC": roc_auc_score(g.y, g.p), "recall": g[g.y == 1].pred.mean(),
                 "flagged_share": g.pred.mean(), "precision": g[g.pred == 1].y.mean()})
fb = pd.cut(te.fico, bins["fico"][0], labels=bins["fico"][1], right=False)
for b, g in te.groupby(fb, observed=True):
    by_v.append({"group": f"FICO {b}", "loans": len(g), "default_rate": g.y.mean(),
                 "AUC": roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else np.nan,
                 "recall": g[g.y == 1].pred.mean(), "flagged_share": g.pred.mean(),
                 "precision": g[g.pred == 1].y.mean() if g.pred.sum() else np.nan})
pd.DataFrame(by_v).to_csv(os.path.join(OUT, "performance_by_segment.csv"), index=False)

cols = ["fico", "ltv", "dti", "orig_rate", "orig_upb"]
fn_df, tp_df = te[(te.y == 1) & (te.pred == 0)], te[(te.y == 1) & (te.pred == 1)]
fp_df, tn_df = te[(te.y == 0) & (te.pred == 1)], te[(te.y == 0) & (te.pred == 0)]
err = pd.DataFrame({"missed defaults (FN)": fn_df[cols].median(), "caught defaults (TP)": tp_df[cols].median(),
                    "false alarms (FP)": fp_df[cols].median(), "correct non-defaults (TN)": tn_df[cols].median()})
err.loc["share_purchase"] = [(d.purpose == "P").mean() for d in (fn_df, tp_df, fp_df, tn_df)]
err.loc["share_2006_2008"] = [(d.vintage <= 2008).mean() for d in (fn_df, tp_df, fp_df, tn_df)]
err.loc["share_ended_in_loss"] = [d.default_strict.mean() for d in (fn_df, tp_df, fp_df, tn_df)]
err.to_csv(os.path.join(OUT, "error_analysis.csv"))

# Loss severity of caught vs missed defaults (the dollar view of errors)
summary["loss_caught_vs_missed"] = {
    "caught_realized_loss": float(tp_df.actual_loss.where(tp_df.severity.notna()).sum()),
    "missed_realized_loss": float(fn_df.actual_loss.where(fn_df.severity.notna()).sum())}

# Example predictions: loans near the 5th, 50th, and 99th percentile of predicted risk
ex_rows = []
for q in [0.05, 0.50, 0.99]:
    target = np.quantile(te.p, q)
    i = (te.p - target).abs().idxmin()
    r = te.loc[i]
    ex_rows.append({"risk_percentile": int(q * 100), "predicted_score": r.p, "fico": r.fico, "ltv": r.ltv,
                    "dti": r.dti, "orig_rate": r.orig_rate, "purpose": r.purpose, "state": r.state,
                    "vintage": r.vintage, "num_borrowers": r.num_borrowers, "actual_default": int(r.y)})
pd.DataFrame(ex_rows).to_csv(os.path.join(OUT, "example_predictions.csv"), index=False)

# Mean predicted score vs actual default rate by decile (calibration check)
te["decile"] = pd.qcut(te.p, 10, labels=False, duplicates="drop") + 1
cal = te.groupby("decile").agg(loans=("y", "size"), mean_score=("p", "mean"), actual_rate=("y", "mean")).reset_index()
cal.to_csv(os.path.join(OUT, "risk_deciles.csv"), index=False)
summary["top_decile_share_of_defaults"] = float(te[te.decile == 10].y.sum() / te.y.sum())
summary["top_two_deciles_share_of_defaults"] = float(te[te.decile >= 9].y.sum() / te.y.sum())

# ---------------------------------------------------------------- 7. Interpretation of tuned models
names = lr_best.named_steps["pre"].get_feature_names_out()
coefs = lr_best.named_steps["clf"].coef_[0]
odds = pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
odds.reindex(odds.coef.abs().sort_values(ascending=False).index) \
    .to_csv(os.path.join(OUT, "tuned_logit_odds_ratios.csv"), index=False)
Xs = X_te.sample(20000, random_state=0)
ys = pd.Series(y_te, index=X_te.index).loc[Xs.index]
pi = permutation_importance(gb_best, Xs, ys, scoring="roc_auc", n_repeats=5, random_state=0)
pd.DataFrame({"feature": Xs.columns, "auc_drop": pi.importances_mean, "auc_drop_std": pi.importances_std}) \
    .sort_values("auc_drop", ascending=False).to_csv(os.path.join(OUT, "tuned_gb_importance.csv"), index=False)

summary["results"] = res
summary["runtime_seconds"] = round(time.time() - t0)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
