# Data

The raw data is **not included** in this repository. Freddie Mac provides the
Single-Family Loan-Level Dataset free of charge but requires registration, and its
terms restrict redistribution. Only aggregated results (in `results/`) are committed.

## How to get the files

1. Register and sign in to Freddie Mac's Clarity Data Intelligence portal:
   https://capitalmarkets.freddiemac.com/clarity
2. Open the SFLLD download page:
   https://claritydownload.fmapps.freddiemac.com/CRT/#/sflld
3. In the **Sample File** column, download `sample_2006.zip`, `sample_2007.zip`,
   `sample_2008.zip`, `sample_2015.zip`, and `sample_2019.zip`.
4. Unzip them into this folder (`data/raw/`). Each year contains:
   - `sample_orig_YYYY.txt`: one row per loan (50,000 loans), origination attributes
   - `sample_svcg_YYYY.txt` (may also be named `sample_perf_YYYY.txt`): one row per
     loan per month, performance history. The code accepts either name.

The files are pipe-delimited with no header row. Column definitions are in Freddie
Mac's *Single-Family Loan-Level Dataset General User Guide* (July 2026):
https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf

The dataset is refreshed periodically, so a later download may have a different
performance cutoff date and slightly different results. The results in this repo
use files downloaded in September 2026 (Release 47), with performance data through
March 31, 2026.
