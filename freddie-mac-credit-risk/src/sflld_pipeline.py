"""
Core pipeline for the Freddie Mac Single-Family Loan-Level Dataset (SFLLD) samples:
loading, loan-level default/loss targets, PD models, and aggregated exports.

Column positions follow the Freddie Mac SFLLD General User Guide (July 2026, Release 47:
31 origination columns and 35 performance columns) and were checked against the actual
sample files. Release 47 moved Servicer Name and the MI Cancellation Indicator from the
origination file to the performance file and added VantageScore 4.0.

This module is imported by run_analysis.py, which is the main entry point.
Expects sample_orig_YYYY.txt and sample_svcg_YYYY.txt (or sample_perf_YYYY.txt)
in DATA_DIR (default: data/raw/).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(REPO_ROOT, "data", "raw"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(REPO_ROOT, "results"))
CACHE_DIR = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Origination columns (0-indexed positions from the user guide, cols 1-24)
ORIG_COLS = {
    0: "fico", 1: "first_pay_date", 2: "first_time_buyer", 4: "msa",
    5: "mi_pct", 6: "num_units", 7: "occupancy", 8: "cltv", 9: "dti",
    10: "orig_upb", 11: "ltv", 12: "orig_rate", 13: "channel",
    16: "state", 17: "property_type", 19: "loan_id", 20: "purpose",
    21: "orig_term", 22: "num_borrowers", 23: "seller",
}
# Performance columns (0-indexed, cols 1-28)
PERF_COLS = {
    0: "loan_id", 1: "period", 3: "dlq_status", 8: "zb_code",
    21: "actual_loss", 26: "zb_removal_upb", 29: "assist_status",
}

DEFAULT_ZB = {"02", "03", "09"}   # third-party sale, short sale/charge-off, REO disposition
LOSS_ZB = {"02", "03", "09", "15"}  # codes for which the guide computes Actual Loss


def load_orig(year):
    path = os.path.join(DATA_DIR, f"sample_orig_{year}.txt")
    df = pd.read_csv(path, sep="|", header=None, usecols=list(ORIG_COLS),
                     dtype=str, keep_default_na=False)
    df = df.rename(columns=ORIG_COLS)
    num = ["fico", "mi_pct", "num_units", "cltv", "dti", "orig_upb", "ltv",
           "orig_rate", "orig_term", "num_borrowers"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Sentinel "not available" codes from the user guide
    df.loc[df.fico == 9999, "fico"] = np.nan
    df.loc[df.dti == 999, "dti"] = np.nan
    df.loc[df.ltv == 999, "ltv"] = np.nan
    df.loc[df.cltv == 999, "cltv"] = np.nan
    df.loc[df.mi_pct == 999, "mi_pct"] = np.nan
    df.loc[df.num_borrowers == 99, "num_borrowers"] = np.nan
    df["vintage"] = int(year)
    return df


def load_perf_summary(year):
    """Roll monthly records up to one row per loan."""
    path = os.path.join(DATA_DIR, f"sample_svcg_{year}.txt")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, f"sample_perf_{year}.txt")
    parts = []
    for chunk in pd.read_csv(path, sep="|", header=None, usecols=list(PERF_COLS),
                             dtype=str, keep_default_na=False, chunksize=1_000_000):
        chunk = chunk.rename(columns=PERF_COLS)
        dlq = chunk.dlq_status
        is_ra = dlq == "RA"
        dlq_num = pd.to_numeric(dlq.where(~is_ra), errors="coerce")
        raw_d90 = (dlq_num >= 3) | is_ra
        # Exclude months where the borrower was in a forbearance plan (e.g. COVID relief);
        # field is populated from Jan 2014 onward per the user guide.
        chunk["d90_raw"] = raw_d90
        chunk["d90"] = raw_d90 & (chunk.assist_status != "F")
        chunk["actual_loss"] = pd.to_numeric(chunk.actual_loss, errors="coerce")
        chunk["zb_removal_upb"] = pd.to_numeric(chunk.zb_removal_upb, errors="coerce")
        chunk["period"] = pd.to_numeric(chunk.period, errors="coerce")
        g = chunk.groupby("loan_id").agg(
            ever_d90=("d90", "max"),
            ever_d90_raw=("d90_raw", "max"),
            zb_code=("zb_code", lambda s: s[s != ""].iloc[-1] if (s != "").any() else ""),
            actual_loss=("actual_loss", "max"),
            zb_removal_upb=("zb_removal_upb", "max"),
            last_period=("period", "max"),
        )
        parts.append(g)
    p = pd.concat(parts)
    # A loan can straddle chunk boundaries; combine again
    p = p.groupby(level=0).agg(
        ever_d90=("ever_d90", "max"),
        ever_d90_raw=("ever_d90_raw", "max"),
        zb_code=("zb_code", lambda s: s[s != ""].iloc[-1] if (s != "").any() else ""),
        actual_loss=("actual_loss", "max"),
        zb_removal_upb=("zb_removal_upb", "max"),
        last_period=("last_period", "max"),
    )
    return p.reset_index()


def build_dataset(years):
    frames = []
    for y in years:
        o = load_orig(y)
        p = load_perf_summary(y)
        frames.append(o.merge(p, on="loan_id", how="left"))
    df = pd.concat(frames, ignore_index=True)
    df["default"] = (df.ever_d90.fillna(False) | df.zb_code.isin(DEFAULT_ZB)).astype(int)
    # Comparison definitions
    df["default_naive"] = (df.ever_d90_raw.fillna(False) | df.zb_code.isin(DEFAULT_ZB)).astype(int)
    df["default_strict"] = df.zb_code.isin(LOSS_ZB).astype(int)  # ended in a credit-loss event
    df["outcome"] = np.select(
        [df.default == 1, df.zb_code == "01"], ["Default", "Prepaid"], "Active/Other")
    has_loss = df.zb_code.isin(LOSS_ZB) & df.actual_loss.notna() & (df.zb_removal_upb > 0)
    df["severity"] = np.where(has_loss, df.actual_loss / df.zb_removal_upb, np.nan)
    return df


NUM_FEATS = ["fico", "ltv", "dti", "orig_rate", "orig_upb", "orig_term",
             "num_borrowers", "mi_pct"]
CAT_FEATS = ["purpose", "occupancy", "channel", "property_type",
             "first_time_buyer", "state"]


def train_models(df, use_vintage=True):
    X = df[NUM_FEATS + CAT_FEATS].copy()
    y = df["default"].values
    if use_vintage and df.vintage.nunique() > 1:
        X["vintage"] = df.vintage.astype(str)
        cats = CAT_FEATS + ["vintage"]
    else:
        cats = CAT_FEATS
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pre_lr = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cats),
    ])
    lr = Pipeline([("pre", pre_lr),
                   ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr.fit(X_tr, y_tr)

    pre_gb = ColumnTransformer([
        ("num", "passthrough", NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                              sparse_output=False), cats),
    ])
    gb = Pipeline([("pre", pre_gb),
                   ("clf", HistGradientBoostingClassifier(
                       max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                       l2_regularization=1.0, class_weight="balanced",
                       early_stopping=True, random_state=42))])
    gb.fit(X_tr, y_tr)

    results, rocs = {}, []
    for name, m in [("Logistic Regression", lr), ("Gradient Boosting", gb)]:
        p = m.predict_proba(X_te)[:, 1]
        results[name] = {"AUC": roc_auc_score(y_te, p),
                         "PR_AUC": average_precision_score(y_te, p)}
        fpr, tpr, _ = roc_curve(y_te, p)
        idx = np.linspace(0, len(fpr) - 1, 200).astype(int)
        rocs.append(pd.DataFrame({"model": name, "fpr": fpr[idx], "tpr": tpr[idx]}))
    return lr, gb, results, pd.concat(rocs), (X_te, y_te)


def tableau_exports(df, rocs):
    """Write aggregated (not loan-level) CSVs used for the Tableau charts."""
    d = df.copy()
    # Left-closed bins so each label matches its range exactly (e.g. 620 falls in "620-659")
    d["fico_band"] = pd.cut(d.fico, [300, 620, 660, 700, 740, 780, 851], right=False,
                            labels=["<620", "620-659", "660-699", "700-739", "740-779", "780+"])
    d["ltv_band"] = pd.cut(d.ltv, [0, 60, 70, 80, 90, 95, 200],
                           labels=["<=60", "61-70", "71-80", "81-90", "91-95", ">95"])
    heat = (d.groupby(["vintage", "fico_band", "ltv_band"], observed=True)
              .agg(loans=("default", "size"), default_rate=("default", "mean"),
                   avg_severity=("severity", "mean")).reset_index())
    heat = heat[heat.loans >= 30]  # suppress tiny cells
    heat.to_csv(os.path.join(OUT_DIR, "tableau_fico_ltv_heatmap.csv"), index=False)

    state = (d.groupby(["vintage", "state"])
               .agg(loans=("default", "size"), default_rate=("default", "mean"),
                    avg_severity=("severity", "mean")).reset_index())
    state.to_csv(os.path.join(OUT_DIR, "tableau_state.csv"), index=False)

    rocs.to_csv(os.path.join(OUT_DIR, "tableau_roc.csv"), index=False)
