# Universe Specification and Paper Crosscheck

This note pins down the stock-universe rules used in this repo and maps them to the original Krauss et al. (2017) paper.

It is meant to answer a narrow question:

- What does the paper say about universe construction?
- What does the current code actually do?
- Which parts are exact matches, and which parts are interpretation choices?

## Bottom line

The repo's primary universe logic is a **monthly-updated, no-lookahead S&P 500 eligibility rule**:

- determine index membership at the **last calendar day of month M**
- use that membership set for **all trading days in month M+1**
- further restrict the usable panel by **return availability**, **feature availability**, and for labels **next-day eligibility**

This matches the paper's clearest universe statement in Section 3.1:

- obtain all month end constituent lists
- consolidate them into a binary matrix
- indicate whether a stock is a constituent in the **subsequent month**

The main ambiguity in the paper is Section 4.1, where `n_i` is defined as the number of stocks in the S&P 500 **at the end of the training period** with **full price information available**. That wording can support a frozen-per-period interpretation. The repo keeps that as an explicit alternative, but not as the primary pipeline.

## Paper Statements

The key universe statements in the paper are:

### Section 3.1 Data

The paper says the authors:

- obtain all month end constituent lists for the S&P 500 from Thomson Reuters Datastream from December 1989 to September 2015
- consolidate these lists into a binary matrix indicating whether a stock is a constituent of the index in the **subsequent month**
- download daily total return indices from January 1990 to October 2015 for **all stocks having ever been a constituent**

This is the clearest and least ambiguous universe definition in the paper.

### Section 4.1 Generation of training and trading sets

The paper also says:

- each study period contains a 750-day training period and a subsequent 250-day trading period
- `n_i` denotes the number of stocks in the S&P 500 at the end of the training period of study period `i`, having full price information available
- the stocks considered in the 23 batches are **time-varying**, depending on **index constituency** and **full data availability**

This is where the ambiguity comes from:

- "at the end of the training period" sounds like a frozen universe
- "time-varying" sounds like membership continues to evolve
- Section 3.1's "subsequent month" language strongly supports monthly updating

## Current CRSP Implementation

The primary CRSP universe logic is implemented in [src/krauss/data/universe.py](/Users/ariusmak/Desktop/krauss/src/krauss/data/universe.py).

### Step 1: Raw membership source

Source table:

- `crsp.dsp500list`

Fetched by:

- [src/krauss/data/wrds_extract.py](/Users/ariusmak/Desktop/krauss/src/krauss/data/wrds_extract.py)
- [scripts/build_data.py](/Users/ariusmak/Desktop/krauss/scripts/build_data.py)

The raw file contains S&P 500 membership spells with:

- `permno`
- `start`
- `ending`

### Step 2: Month-end membership snapshot

Implemented in [src/krauss/data/universe.py](/Users/ariusmak/Desktop/krauss/src/krauss/data/universe.py):25.

For each month-end `me_date`, a stock is considered a constituent if:

- `start <= me_date`
- and `ending >= me_date` or `ending is null`

This exactly recreates a month-end constituent list from spell data.

### Step 3: Shift snapshot to the subsequent month

Implemented in [src/krauss/data/universe.py](/Users/ariusmak/Desktop/krauss/src/krauss/data/universe.py):67.

Each month-end snapshot is assigned to:

- `effective_month = month_end + 1 month`

So:

- January 31 membership governs February
- February 28 membership governs March

This is the repo's core no-lookahead rule.

### Step 4: Expand to daily eligibility

Implemented in [src/krauss/data/universe.py](/Users/ariusmak/Desktop/krauss/src/krauss/data/universe.py):111.

Each trading date is mapped to its calendar month, then joined to the corresponding `effective_month` membership set.

This produces:

- `data/processed/universe_daily.parquet`

Important: this file is an **eligibility panel**, not yet a final tradable panel.

## Effective Tradable Universe in the CRSP Pipeline

The actual stock set used by models is narrower than `universe_daily`.

### Layer 1: Membership eligibility

A stock must be in `universe_daily` for date `t`.

### Layer 2: Return availability

The stock must also have a return observation in the cleaned return panel from:

- [src/krauss/data/prices_returns.py](/Users/ariusmak/Desktop/krauss/src/krauss/data/prices_returns.py)

This includes delisting-return adjustment.

### Layer 3: Feature availability

The stock must have all 31 lagged-return features from:

- [src/krauss/data/features.py](/Users/ariusmak/Desktop/krauss/src/krauss/data/features.py)

Because the longest lookback is 240 trading days, early history is dropped for each stock.

This is the repo's operational interpretation of "full price information available."

### Layer 4: Label availability

Labels are built in:

- [src/krauss/data/labels.py](/Users/ariusmak/Desktop/krauss/src/krauss/data/labels.py)

For a labeled row at date `t`, the stock must:

- be eligible on `t`
- have a realized return on `t+1`
- also be eligible on `t+1`

The next-day cross-sectional median is then computed over the next-day eligible set.

This makes the label universe stricter than the feature universe.

### Layer 5: Modeling panel

The actual train/trade panel is built by inner-joining:

- features
- labels
- daily eligibility

in [scripts/run_phase1.py](/Users/ariusmak/Desktop/krauss/scripts/run_phase1.py):66 and [scripts/run_phase2.py](/Users/ariusmak/Desktop/krauss/scripts/run_phase2.py):46.

## Current Datastream Implementation

The Datastream pipeline lives in [scripts/build_data_datastream.py](/Users/ariusmak/Desktop/krauss/scripts/build_data_datastream.py).

### Exact paper matches

It follows the paper more directly than the CRSP pipeline:

- membership source is Datastream `ds2constmth`
- identifier is Datastream `infocode`
- returns are computed from Datastream return index `RI`
- month-end snapshots are shifted to the subsequent month
- returns are downloaded for all stocks that ever belonged to the index

This is the cleanest implementation of Section 3.1 in the repo.

### Important caveat

The main builder script writes the unfiltered `ds_*` parquet files.

The paper-parity workflow described elsewhere in the repo relies on saved `*_usonly.parquet` files and on:

- [scripts/test_datastream_h2o.py](/Users/ariusmak/Desktop/krauss/scripts/test_datastream_h2o.py)

So the **artifacts** reflect the US-trading-calendar-filtered interpretation, but that filtering is not encoded in the primary Datastream build script itself.

## Frozen-Per-Period Alternative

The frozen alternative is implemented in:

- [src/krauss/data/universe_frozen.py](/Users/ariusmak/Desktop/krauss/src/krauss/data/universe_frozen.py)
- [scripts/test_frozen_universe.py](/Users/ariusmak/Desktop/krauss/scripts/test_frozen_universe.py)

This interpretation reads Section 4.1 literally:

- take the S&P 500 membership set at the end of the training period
- keep that stock set fixed through the trade period
- only remove stocks when they no longer have return observations

This is a defensible reading of the paper's `n_i` sentence, but it is not the primary implementation because it fits Section 3.1 less well.

## Crosscheck Verdict

### Exact or near-exact matches to the paper

- month-end constituent lists drive the universe
- month-end membership is used for the subsequent month
- ever-members are used for the return download universe
- the stock set is time-varying in the primary pipeline
- effective usable observations depend on both constituency and data availability

### Interpretation choices in the repo

- "full price information available" is implemented operationally through return availability, 240-day feature availability, and next-day label eligibility
- the primary pipeline chooses monthly updating over frozen-per-period membership
- the label universe requires next-day eligibility, which is reasonable but more explicit than the paper

### Known caveat

- Datastream US-calendar filtering is supported by saved artifacts and downstream scripts, but not fully encoded in the main Datastream builder

## Why the Current Primary Implementation Is Still Plausible

Empirically, the current monthly-updated implementation appears to be the most paper-consistent version in this repo:

- alternative universe interpretations produced noticeably different universe counts
- those differences were too large to explain away purely by CRSP versus Datastream source differences
- alternative interpretations also produced materially different return behavior

That does not make the monthly-updated interpretation logically certain, but it does make it the strongest combined reading of:

- the paper's Section 3.1 wording
- the paper's reported stock counts
- the repo's reproduction experiments

## Practical Definition Used Going Forward

When we refer to "the universe" in this repo, the safest precise meaning is:

`month-end S&P 500 membership shifted to the subsequent month, then filtered by actual data availability required for returns, features, labels, and trade execution`

That is the operative definition used by the main CRSP and Datastream reproduction pipelines.
