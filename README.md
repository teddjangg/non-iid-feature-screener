# iid-screener

A lightweight, domain-knowledge-free tool for detecting potential IID assumption violations at the feature level.

## Quick Start

```python
import pandas as pd
from screen_non_iid import screen_non_iid_features

X = data.drop(["target", "id"], axis=1)
y = data["target"]

result = screen_non_iid_features(X, y, n_splits=5, ratio_threshold=2.0)
print(result.to_string(index=False))
```

Output:

```
                 feature  pearson_ratio  spearman_ratio
    data_channel_is_tech       7.268344        1.065882
                  LDA_04       5.127492        0.949443
               kw_max_min       2.435606        0.680475
                   ...
```

---

## Motivation

In practice, datasets are not always explicitly labeled as time series. A dataset may look cross-sectional on the surface, yet contain features with latent temporal structure — features whose relationship with the target drifts over time.

When such features exist and a model is trained under the IID assumption, future data can subtly leak into training folds during cross-validation. This creates a **mini data leakage** problem: the model appears to generalize better than it actually does, because validation folds unknowingly contain information from the future.

**The goal of this project is not interpretation or performance improvement.** It is purely a **first-pass screening tool** — to flag features that may violate the IID assumption, without any domain knowledge required. What you do with those features afterward is up to you.

---

## How It Works

This project builds on the core insight from my earlier project [validation-instability-analysis](https://github.com/teddjangg/validation-instability-analysis):

> When a dataset contains latent temporal structure, KFold and TimeSeriesSplit produce systematically different validation behavior.

Both KFold and TimeSeriesSplit split data into folds. For each fold, we compute the **correlation gap** between train and test sets per feature:

```
gap = corr(feature, target)_train - corr(feature, target)_test
```

We then look at the **variance of this gap across folds**:

- **KFold** ignores row order → gap variance stays low even for non-IID features
- **TimeSeriesSplit** respects row order → gap variance inflates for non-IID features

If the data were truly IID, both splitters should produce stable, low-variance gaps. When TSSplit variance is significantly higher than KFold variance, it signals that **row order matters** for that feature — a sign of latent temporal structure and potential IID violation.

---

## Scoring

We test with two correlation measures:

- **Pearson** — captures linear relationships between feature and target
- **Spearman** — rank-based, more robust to non-linear relationships

For each feature, we compute:

```
pearson_ratio  = TSSplit gap variance / KFold gap variance  (Pearson)
spearman_ratio = TSSplit gap variance / KFold gap variance  (Spearman)
```

A feature is flagged as a **potential IID violator** if either ratio exceeds the threshold (default: 2.0):

```python
is_flagged = (pearson_ratio > 2.0) | (spearman_ratio > 2.0)
```

- `pearson_ratio` high only → linear relationship is unstable across folds
- `spearman_ratio` high only → rank-based relationship is unstable across folds
- Both high → strong signal from both perspectives

The threshold of 2.0 is intentionally conservative. You can adjust it:

```python
result = screen_non_iid_features(X, y, ratio_threshold=3.0)
```

---

## Experiments

### Experiment 1: Online News Popularity (cross-sectional)

A dataset originally designed for cross-sectional use. We expect weak temporal signal.

| feature | pearson_ratio | spearman_ratio |
|---|---|---|
| data_channel_is_tech | 7.27 | 1.07 |
| LDA_04 | 5.13 | 0.95 |
| kw_max_min | 2.44 | 0.68 |
| kw_avg_min | 2.23 | 2.19 |
| weekday_is_wednesday | 2.13 | 3.47 |
| ... | ... | ... |

![Online News Scatter](results/news_scatter.png)

Keyword-related features (`kw_*`) and topic features (`LDA_04`, `data_channel_is_tech`) are flagged. This makes intuitive sense — keyword popularity and content categories tend to drift over time as internet trends evolve.

---

### Experiment 2: Air Quality UCI (time series)

A genuine time series dataset. We expect strong temporal signal across many features.

| feature | pearson_ratio | spearman_ratio |
|---|---|---|
| RH | 4.95 | 0.56 |
| PT08.S4(NO2) | 4.63 | 4.13 |
| T | 4.21 | 2.00 |
| AH | 3.08 | 2.37 |
| C6H6(GT) | 2.71 | 2.53 |
| ... | ... | ... |

![Air Quality Scatter](results/airquality_scatter.png)

Temperature, humidity, and pollutant sensor features are flagged — all variables with strong seasonal patterns. The screener captures these without any time column or domain knowledge.

---

### Experiment 3: Wine (IID baseline)

A purely IID dataset. We expect nothing to be flagged.

```
Empty DataFrame
Columns: [feature, pearson_ratio, spearman_ratio]
```

No features are flagged. This confirms the screener does not produce false positives on genuinely IID data.

---

## Limitations

- The ratio threshold (default: 2.0) is heuristic. There is no statistical guarantee that a flagged feature truly has temporal structure — it is a first-pass signal only.
- Results depend on row order. If the data has no meaningful ordering, results may be noisy.
- With few folds (default: 5), variance estimates are based on limited samples. Features with very sparse non-null values per fold may be excluded automatically (`min_valid_folds=4`).

---

## Related

- [validation-instability-analysis](https://github.com/teddjangg/validation-instability-analysis) — the project this screener builds on
