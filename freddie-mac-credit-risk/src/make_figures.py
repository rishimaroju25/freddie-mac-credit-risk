"""
Render the static figures from the aggregated CSVs in results/.
Usage (from the repo root):
    python src/make_figures.py              # light theme -> figures/
    THEME=dark python src/make_figures.py   # dark theme  -> figures/dark/ (for the portfolio site)
Needs only results/, not the raw Freddie Mac files.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DARK = os.environ.get("THEME", "light") == "dark"
FIG = os.path.join(ROOT, "figures", "dark") if DARK else os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

if DARK:  # matched to the portfolio site's dark palette
    SURFACE, INK, INK_2, GRID = "#11141a", "#eceef1", "#9aa1ab", "#262a32"
    BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
    # on a dark surface, magnitude reads as increasing lightness
    SEQ_BLUE = ["#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]
else:
    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def text_on(hex_color):
    """White or near-black text, whichever reads on the given fill."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.45 else "#ffffff"
VINTAGES = ["2006", "2007", "2008", "2015", "2019"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.axisbelow": True,
})


def pct(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def title(fig, main, sub):
    fig.text(0.02, 0.965, main, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, sub, fontsize=10.5, color=INK_2, va="top")


# 1. Default definitions ---------------------------------------------------------
d = pd.read_csv(os.path.join(RES, "tableau_default_definitions.csv"))
d["vintage"] = d.vintage.astype(str)
series = [
    ("Naive 90+ DPD (incl. forbearance)", "Naive: 90+ days delinquent (incl. forbearance)", ORANGE),
    ("90+ DPD excl. forbearance (model target)", "Model target: excl. forbearance months", BLUE),
    ("Strict: ended in credit-loss event", "Strict: loan ended in a credit loss", AQUA),
]
fig, ax = plt.subplots(figsize=(10, 5.6))
fig.subplots_adjust(left=0.08, right=0.80, top=0.80, bottom=0.10)
x = np.arange(len(VINTAGES))
for key, label, color in series:
    s = d[d.definition == key].set_index("vintage").loc[VINTAGES, "rate"].values
    ax.plot(x, s, color=color, lw=2, solid_capstyle="round", label=label, zorder=3)
    ax.scatter(x, s, s=44, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.annotate(f"{s[-1]:.1%}", (x[-1], s[-1]), xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=10.5, color=INK, fontweight="bold")
ax.text(4, 0.078, "Gap in 2019 =\nCOVID forbearance", ha="center",
        fontsize=10, color=INK_2)
ax.set_xticks(x, VINTAGES)
ax.set_xlabel("Origination vintage (sampled years; 2009-2014 not included)")
ax.set_ylim(0, 0.18)
pct(ax)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", frameon=False, fontsize=10)
title(fig, "How the default definition changes the story",
      "Share of loans meeting each definition, by origination year (50,000-loan sample per year)")
fig.savefig(os.path.join(FIG, "default_definitions.png"), dpi=160)
plt.close(fig)

# 2. FICO x LTV heatmap (all vintages pooled, loan-weighted) -------------------------
h = pd.read_csv(os.path.join(RES, "tableau_fico_ltv_heatmap.csv"))
h["defaults"] = h.loans * h.default_rate
g = h.groupby(["fico_band", "ltv_band"])[["loans", "defaults"]].sum()
g["rate"] = g.defaults / g.loans
fico_order = ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]
ltv_order = ["<=60", "61-70", "71-80", "81-90", "91-95", ">95"]
m = g.rate.unstack().reindex(index=fico_order, columns=ltv_order)
cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
fig, ax = plt.subplots(figsize=(9.5, 6.2))
fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.10)
vmax = float(np.nanmax(m.values))
ax.imshow(m.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        v = m.values[i, j]
        if np.isnan(v):
            continue
        ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=11,
                color=text_on(matplotlib.colors.to_hex(cmap(v / vmax))))
ax.set_xticks(range(len(ltv_order)), ltv_order)
ax.set_yticks(range(len(fico_order)), fico_order)
ax.set_xlabel("Original loan-to-value (%)")
ax.set_ylabel("Credit score (FICO)")
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, len(ltv_order)), minor=True)
ax.set_yticks(np.arange(-0.5, len(fico_order)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
title(fig, "Default rate by credit score and loan-to-value",
      "All five vintages pooled (250,000 loans). Each cell is the default rate within that group.")
fig.savefig(os.path.join(FIG, "fico_ltv_heatmap.png"), dpi=160)
plt.close(fig)

# 3. CRT-style tiers: default rate vs loss severity --------------------------------
t = pd.read_csv(os.path.join(RES, "tableau_crt_tiers.csv"))
t["vintage"] = t.vintage.astype(str)
tiers = [("High LTV 81-97 (HQA-style)", "High LTV 81-97 (HQA-style)", BLUE),
         ("Low LTV 61-80 (DNA-style)", "Low LTV 61-80 (DNA-style)", ORANGE)]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.12, wspace=0.18)
w = 0.36
for ax, col, name in [(axes[0], "pd_rate", "Default rate"),
                      (axes[1], "lgd", "Median loss severity (share of balance lost)")]:
    for k, (key, label, color) in enumerate(tiers):
        s = t[t.crt_tier == key].set_index("vintage").loc[VINTAGES, col].values
        pos = np.arange(len(VINTAGES)) + (k - 0.5) * (w + 0.03)
        ax.bar(pos, s, width=w, color=color, label=label, zorder=3)
    ax.set_xticks(np.arange(len(VINTAGES)), VINTAGES)
    ax.set_title(name, loc="left", fontsize=11.5, color=INK, pad=10)
    ax.grid(axis="x", visible=False)
    pct(ax)
axes[1].text(3.5, 0.27, "Only 134 and 46 loss\nevents in 2015 and 2019",
             ha="center", fontsize=9.5, color=INK_2)
axes[0].legend(loc="upper right", frameon=False, fontsize=10)
title(fig, "High-LTV loans default more often but lose less per default",
      "CRT-style LTV tiers by vintage. Severity = median actual loss / balance at default, loans with disclosed losses.")
fig.savefig(os.path.join(FIG, "crt_tiers.png"), dpi=160)
plt.close(fig)
print("wrote", sorted(os.listdir(FIG)))

# ------------------------------------------------------------------------------
# Figures from src/model_evaluation.py (skipped if that script has not been run)
ME = os.path.join(RES, "model_evaluation")
if os.path.exists(os.path.join(ME, "summary.json")):
    import json
    S = json.load(open(os.path.join(ME, "summary.json")))

    # 4. Default rate by key feature (exploration)
    b = pd.read_csv(os.path.join(ME, "default_rate_by_band.csv"))
    names = {"fico": "Credit score (FICO)", "ltv": "Loan-to-value (%)",
             "dti": "Debt-to-income (%)", "orig_rate": "Interest rate"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.08, hspace=0.45, wspace=0.18)
    top = b.default_rate.max() * 1.18
    for ax, (f, label) in zip(axes.flat, names.items()):
        g = b[b.feature == f]
        xs = np.arange(len(g))
        ax.bar(xs, g.default_rate, width=0.6, color=BLUE, zorder=3)
        for xi, v in zip(xs, g.default_rate):
            ax.text(xi, v + top * 0.015, f"{v:.1%}", ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(xs, g.band)
        ax.set_ylim(0, top)
        ax.set_title(label, loc="left", fontsize=11.5, color=INK, pad=8)
        ax.grid(axis="x", visible=False)
        pct(ax)
    title(fig, "Default risk rises with lower credit scores, higher leverage, and higher rates",
          f"Default rate by band, all five vintages (250,000 loans; overall rate {S['default_rate_overall']:.1%})")
    fig.savefig(os.path.join(FIG, "eda_default_by_feature.png"), dpi=160)
    plt.close(fig)

    # 5. Outcomes (class imbalance)
    o = pd.Series(S["outcomes"]).reindex(["Prepaid", "Active/Other", "Default"])
    labels = {"Prepaid": "Paid off", "Active/Other": "Still active or other ending", "Default": "Default (target = 1)"}
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    fig.subplots_adjust(left=0.27, right=0.9, top=0.72, bottom=0.11)
    ys = np.arange(len(o))[::-1]
    ax.barh(ys, o.values / o.sum(), height=0.55, color=[INK_2, INK_2, ORANGE], zorder=3)
    for yi, v in zip(ys, o.values):
        ax.text(v / o.sum() + 0.01, yi, f"{v / o.sum():.1%}  ({v:,})", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(ys, [labels[k] for k in o.index])
    ax.set_xlim(0, 1)
    pct(ax, "x")
    ax.grid(axis="y", visible=False)
    title(fig, "Only about 1 in 11 loans defaulted", "How the 250,000 sampled loans ended up (performance through March 2026)")
    fig.savefig(os.path.join(FIG, "outcomes.png"), dpi=160)
    plt.close(fig)

    # 6. Model comparison vs baselines
    mc = pd.read_csv(os.path.join(ME, "model_comparison.csv"), index_col=0)
    order = ["Baseline: predict no default", "Baseline: credit score only",
             "Logistic Regression (tuned)", "Gradient Boosting (tuned)"]
    short = ["Predict no default", "Credit score only", "Logistic regression", "Gradient boosting"]
    colors = [INK_2, INK_2, BLUE, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.72, bottom=0.14, wspace=0.55)
    for ax, col, name in [(axes[0], "AUC", "AUC (0.5 = coin flip, 1.0 = perfect)"),
                          (axes[1], "PR_AUC", "PR-AUC (focus on finding defaults)")]:
        v = mc.loc[order, col].values
        ys = np.arange(len(order))[::-1]
        ax.barh(ys, v, height=0.55, color=colors, zorder=3)
        for yi, val in zip(ys, v):
            ax.text(val + 0.01, yi, f"{val:.3f}", va="center", fontsize=10.5, color=INK)
        ax.set_yticks(ys, short)
        ax.set_xlim(0, 1.0 if col == "AUC" else 0.6)
        ax.set_title(name, loc="left", fontsize=11, color=INK, pad=8)
        ax.grid(axis="y", visible=False)
    title(fig, "Both models beat the baselines; gradient boosting edges out logistic regression",
          "Held-out test set of 50,000 loans. Gray = baselines, blue = tuned models.")
    fig.savefig(os.path.join(FIG, "model_comparison.png"), dpi=160)
    plt.close(fig)

    # 7. Confusion matrix
    cm = S["confusion_matrix"]
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    fig.subplots_adjust(left=0.24, right=0.96, top=0.76, bottom=0.14)
    shade = np.array([[SEQ_BLUE[0], SEQ_BLUE[1]], [SEQ_BLUE[1], SEQ_BLUE[4]]])
    names_cm = [["Correctly cleared", "False alarm"], ["Missed default", "Caught default"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=shade[i, j], ec=SURFACE, lw=3))
            tc = text_on(shade[i, j])
            ax.text(j, i - 0.08, f"{mat[i, j]:,}", ha="center", va="center", fontsize=17,
                    fontweight="bold", color=tc)
            ax.text(j, i + 0.2, names_cm[i][j], ha="center", va="center", fontsize=10.5,
                    color=tc, alpha=0.85)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1], ["Predicted: no default", "Predicted: default"])
    ax.set_yticks([0, 1], ["Actual: no default", "Actual: default"])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    a = S["at_threshold"]
    title(fig, "Gradient boosting at the chosen cutoff",
          f"Catches {a['recall']:.0%} of defaults; {a['precision']:.0%} of flagged loans actually default "
          f"(cutoff {S['threshold']:.2f}, test set)")
    fig.savefig(os.path.join(FIG, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    # 8. Risk deciles
    rd = pd.read_csv(os.path.join(ME, "risk_deciles.csv"))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.14)
    ax.bar(rd.decile, rd.actual_rate, width=0.6, color=[SEQ_BLUE[2]] * 9 + [BLUE], zorder=3)
    for xi, v in zip(rd.decile, rd.actual_rate):
        ax.text(xi, v + 0.006, f"{v:.1%}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(rd.decile, [f"{d}" for d in rd.decile])
    ax.set_xlabel("Predicted risk decile (1 = safest 10% of loans, 10 = riskiest 10%)")
    ax.grid(axis="x", visible=False)
    pct(ax)
    title(fig, "The riskiest 10% of loans hold " + f"{S['top_decile_share_of_defaults']:.0%} of all defaults",
          "Actual default rate by the gradient boosting model's predicted risk decile (test set)")
    fig.savefig(os.path.join(FIG, "risk_deciles.png"), dpi=160)
    plt.close(fig)

    # 9. Feature importance (tuned gradient boosting)
    fi = pd.read_csv(os.path.join(ME, "tuned_gb_importance.csv")).head(10)[::-1]
    nice = {"fico": "Credit score", "orig_rate": "Interest rate", "ltv": "Loan-to-value", "state": "State",
            "num_borrowers": "Number of borrowers", "purpose": "Loan purpose", "dti": "Debt-to-income",
            "vintage": "Origination year", "orig_term": "Loan term", "orig_upb": "Loan amount",
            "channel": "Origination channel", "property_type": "Property type", "occupancy": "Occupancy",
            "first_time_buyer": "First-time buyer", "mi_pct": "Mortgage insurance %"}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.subplots_adjust(left=0.24, right=0.95, top=0.78, bottom=0.12)
    ys = np.arange(len(fi))
    ax.barh(ys, fi.auc_drop, height=0.55, color=BLUE, zorder=3)
    for yi, v in zip(ys, fi.auc_drop):
        ax.text(v + 0.001, yi, f"{v:.3f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(ys, [nice.get(f, f) for f in fi.feature])
    ax.set_xlabel("Drop in AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    title(fig, "Credit score matters most, followed by interest rate and loan-to-value",
          "Permutation importance, tuned gradient boosting, 20,000 test loans (top 10 features)")
    fig.savefig(os.path.join(FIG, "feature_importance.png"), dpi=160)
    plt.close(fig)
    print("wrote evaluation figures")
