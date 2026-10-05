# Mortgage Credit Risk: Default and Loss Across Freddie Mac Vintages

Predicting which mortgages go seriously delinquent, measuring how much is lost when they do,
and testing how much the answer depends on how "default" is defined. Built on a 250,000-loan
sample from Freddie Mac's Single-Family Loan-Level Dataset (50,000 loans per origination year,
2006, 2007, 2008, 2015, and 2019), with about 14 million monthly performance records.

**Tools:** Python (pandas, scikit-learn, matplotlib), Tableau

**Full write-up:** [PORTFOLIO_POST.md](https://github.com/rishimaroju25/freddie-mac-credit-risk/blob/main/freddie-mac-credit-risk/PORTFOLIO_POST.md) covers the problem, data exploration,
preparation, baselines, tuning, evaluation, interpretation, ethics, references, and AI disclosure.

## Key findings

1. **COVID forbearance doubles the naive delinquency rate for 2019 loans.** A standard "ever 90+
   days delinquent" flag puts 2019 originations at 5.6%, above 2015. Excluding months when the
   borrower was in a forbearance plan brings it to 2.8%, in line with 2015. Of the 2019 loans
   flagged by the naive definition, 87% had a forbearance flag, and most later paid off or became
   current.
2. **Default models reach 0.85 AUC, but lose ground on a new lending regime.** Gradient boosting
   (AUC 0.847) edged out logistic regression (AUC 0.842) on a random split. Trained only on
   2006 to 2008 loans and scored on 2015 and 2019 loans, AUC fell to about 0.72 to 0.73.
3. **Credit score, interest rate, and loan-to-value drive risk.** Default rates run from about 1%
   for borrowers with 780+ credit scores and LTV at or below 60%, to about 42% for scores under 620
   with LTV above 95%.
4. **High-LTV loans default more often but lose less per default.** For 2007 originations, loans in
   the 81 to 97 LTV tier defaulted at 27% versus 16% for the 61 to 80 tier, but with a median loss
   severity of 43% versus 58%. This is consistent with mortgage insurance absorbing part of the
   loss on high-LTV loans, but the analysis does not test that directly.

![How the default definition changes the story](https://raw.githubusercontent.com/rishimaroju25/freddie-mac-credit-risk/main/freddie-mac-credit-risk/figures/default_definitions.png)

## Research question

Using only information known when a loan is made, how well can we predict whether a Freddie Mac
mortgage will become seriously delinquent, and how does the answer change with (a) the definition
of default, (b) the lending era, and (c) the LTV risk tiers Freddie Mac uses in its Credit Risk
Transfer (CRT) deals?

## Data

Freddie Mac [Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
(SFLLD), Standard Dataset sample files: a simple random sample of 50,000 loans per origination
year. Each year has an origination file (one row per loan) and a monthly performance file
(one row per loan per month). Files were downloaded in September 2026 (Release 47), with
performance data through March 31, 2026.

| Vintage | Why it's included |
|---|---|
| 2006, 2007 | Housing bubble peak; loans hit the 2008 crash early, giving the most defaults |
| 2008 | Transition year as underwriting tightened mid-crisis |
| 2015 | Post-crisis underwriting with about ten years of history |
| 2019 | Pre-pandemic loans, which exposes the effect of COVID forbearance |

The raw files are not included in this repo. See [`data/README.md`](https://github.com/rishimaroju25/freddie-mac-credit-risk/blob/main/freddie-mac-credit-risk/data/README.md) for how to
download them.

## Method

**Target variable.** A loan is a default (1) if it was ever 90+ days delinquent, reached REO, or
ended in a third-party sale, short sale, charge-off, or REO disposition. Months where the servicer
reported a forbearance plan are excluded from the delinquency test. Two alternative definitions are
reported for comparison:

| Vintage | Naive 90+ days (incl. forbearance) | **Model target (excl. forbearance)** | Strict: ended in credit loss |
|---|---|---|---|
| 2006 | 14.4% | **14.3%** | 8.1% |
| 2007 | 16.5% | **16.4%** | 9.0% |
| 2008 | 9.3% | **9.2%** | 4.3% |
| 2015 | 4.1% | **2.5%** | 0.3% |
| 2019 | 5.6% | **2.8%** | 0.1% |

**Features.** Origination attributes only (to avoid data leakage): credit score, LTV, DTI, interest
rate, loan amount, term, number of borrowers, mortgage insurance %, loan purpose, occupancy,
channel, property type, first-time buyer flag, state, and (random-split models only) vintage.

**Models.** Logistic regression (standardized, median-imputed numeric features plus one-hot
categoricals) and histogram gradient boosting, both with balanced class weights to handle the
roughly 9% default rate.

**Validation.** (1) Stratified random 80/20 split across all vintages. (2) Out-of-time test:
train on 2006 to 2008, score 2015 and 2019 separately, without vintage as a feature.

**Loss severity.** For loans that ended in a loss-bearing termination, severity = Freddie Mac's
disclosed Actual Loss / unpaid balance at termination. Reported as medians by CRT-style LTV tier
(61 to 80, like the DNA series; 81 to 97, like the HQA series).

## Results

### Model performance

Tuned with 3-fold cross-validation on the training set and compared against two baselines
(`src/model_evaluation.py`):

| Model (50,000-loan test set) | AUC | PR-AUC |
|---|---|---|
| Baseline: predict no default | 0.500 | 0.090 |
| Baseline: credit score only | 0.753 | 0.232 |
| Logistic regression (tuned) | 0.842 | 0.365 |
| Gradient boosting (tuned, final model) | 0.848 | 0.390 |

At a cutoff chosen on training data (maximizing F2), gradient boosting catches 73% of test-set
defaults while flagging 24% of loans, and the riskiest 10% of loans hold 44% of all defaults.

Untuned models from `src/run_analysis.py`, including the out-of-time test:

| Evaluation | Logistic regression | Gradient boosting |
|---|---|---|
| Random 80/20 split, AUC | 0.842 | 0.847 |
| Random 80/20 split, PR-AUC | 0.365 | 0.386 |
| Out-of-time, 2015 vintage, AUC | 0.723 | 0.732 |
| Out-of-time, 2019 vintage, AUC | 0.717 | 0.729 |

The random-split AUC benefits from vintage being a feature; the out-of-time results are the more
realistic measure of how a model would perform on new loans. The small gap between the two models
suggests the interpretable logistic model is competitive.

### Drivers of default

Permutation importance for gradient boosting (drop in AUC when the feature is shuffled):
credit score (0.081), interest rate (0.053), LTV (0.035), state (0.031), number of borrowers
(0.017), loan purpose (0.010), DTI (0.010). Interest rate partly reflects the risk the lender
priced in and partly the rate environment of the origination year, so it is not a pure borrower
risk signal. The highest-risk states in the logistic model were Nevada, Florida, and Arizona.

![Default rate by credit score and loan-to-value](https://raw.githubusercontent.com/rishimaroju25/freddie-mac-credit-risk/main/freddie-mac-credit-risk/figures/fico_ltv_heatmap.png)

### Loss by CRT-style LTV tier

![High-LTV loans default more often but lose less per default](https://raw.githubusercontent.com/rishimaroju25/freddie-mac-credit-risk/main/freddie-mac-credit-risk/figures/crt_tiers.png)

Across all 9,940 loss events, the median loss was about 52% of the balance at termination.

## How to read the percentages

Each percentage is the default rate within its own group of loans, not a share of one total,
so they aren't meant to add up to 100. For example, 27.1% in the top-left heatmap cell means
27.1% of loans with a credit score under 620 and LTV of 60% or less defaulted.

## Limitations

- Post-crisis loss severity rests on few loss events (134 in 2015, 46 in 2019).
- About 10% of loss severities fall outside 0% to 100% (201 below 0%, 815 above 100%, out of
  9,940). Values above 100% are mostly small balances where accrued interest and expenses
  exceed the balance. Medians are used so these outliers do not drive the results.
- Five sampled vintages, not a continuous series; 2009 to 2014 are not included.
- The high-LTV / mortgage insurance link is a correlation, not a causal test.
- The dataset is refreshed periodically, so later downloads may produce slightly different numbers.
- The forbearance exclusion relies on the Borrower Assistance Status field, which Freddie Mac
  populates from January 2014 onward.

## Next steps

- Model loss severity (LGD) directly and combine it with PD into expected loss
  (PD × LGD × EAD) by CRT-style tier.
- Add the 2009 to 2014 vintages and house price data to test the regional findings.
- Publish an interactive Tableau dashboard.

## Reproduce

```bash
pip install -r requirements.txt
# put the unzipped sample files in data/raw/ (see data/README.md)
python src/run_analysis.py 2006 2007 2008 2015 2019   # a few minutes; writes results/
python src/model_evaluation.py                         # about 10 minutes; baselines, tuning, evaluation
python src/make_figures.py                             # writes figures/ from results/
```

Random seeds are fixed, and re-running the pipeline reproduced identical files in `results/`.

## Repository structure

```
├── data/
│   └── README.md            how to download the Freddie Mac files (not committed)
├── figures/                 charts used in this README
├── results/                 aggregated outputs (also the Tableau data sources)
│   ├── tableau_default_definitions.csv
│   ├── tableau_fico_ltv_heatmap.csv
│   ├── tableau_state.csv
│   ├── tableau_crt_tiers.csv
│   ├── tableau_out_of_time.csv
│   ├── tableau_roc.csv
│   ├── tableau_gb_importance.csv
│   ├── tableau_logit_odds_ratios.csv
│   └── model_evaluation/    baselines, tuning, confusion matrix, error analysis, data profile
├── PORTFOLIO_POST.md        full project write-up
└── src/
    ├── sflld_pipeline.py    loading, targets, models, aggregated exports
    ├── run_analysis.py      main entry point (builds and caches the loan-level table)
    ├── model_evaluation.py  baselines, CV tuning, final model evaluation
    └── make_figures.py      all charts
```

## Data use

Data provided by Freddie Mac. This project uses the dataset under Freddie Mac's
[terms and conditions](https://capitalmarkets.freddiemac.com/crt/resources/disclosure-guides/disclosure-guides-overview);
only aggregated results are published here. Freddie Mac does not endorse this analysis.

## Author

Rishi · Data Science, UNC Charlotte ·
[GitHub](https://github.com/rishimaroju25) · [LinkedIn](https://www.linkedin.com/in/rishimaroju)

Started as Project Two for DTSC 2301 at UNC Charlotte and extended independently.
