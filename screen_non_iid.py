import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, TimeSeriesSplit

def compute_corr_gap(X, y, splitter):
    """
    Compute per-fold, per-feature Pearson & Spearman train/test correlation gap.
    gap = corr_train - corr_test
    """
    results = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for feature in X.columns:
            x_tr = X_train[feature]
            x_te = X_test[feature]

            if x_tr.nunique() > 1 and x_te.nunique() > 1:
                p_tr = pearsonr(x_tr, y_train)[0]
                p_te = pearsonr(x_te, y_test)[0]
                s_tr = spearmanr(x_tr, y_train)[0]
                s_te = spearmanr(x_te, y_test)[0]
            else:
                p_tr = p_te = s_tr = s_te = np.nan

            results.append({
                "fold":             fold_id,
                "feature":          feature,
                "pearson_train":    p_tr,
                "pearson_test":     p_te,
                "pearson_gap":      p_tr - p_te,
                "pearson_abs_gap":  abs(p_tr - p_te),
                "spearman_train":   s_tr,
                "spearman_test":    s_te,
                "spearman_gap":     s_tr - s_te,
                "spearman_abs_gap": abs(s_tr - s_te),
            })
    return pd.DataFrame(results)


def compute_gap_variance(corr_gap_df, min_valid_folds=4):
    """
    Compute per-feature variance and mean of fold-level correlation gaps.
    Features with fewer than min_valid_folds valid folds are excluded.
    """
    return (
        corr_gap_df
        .groupby("feature")
        .filter(lambda g: g["pearson_gap"].notna().sum() >= min_valid_folds)
        .groupby("feature")
        .agg(
            pearson_gap_var   = ("pearson_gap",  "var"),
            spearman_gap_var  = ("spearman_gap", "var"),
            pearson_gap_mean  = ("pearson_gap",  "mean"),
            spearman_gap_mean = ("spearman_gap", "mean"),
        )
        .reset_index()
        .sort_values("pearson_gap_var", ascending=False)
    )


def make_corr_summary(results_kf, results_ts, features, n_splits=5):
    """
    For flagged features, return a fold-level comparison table of
    KFold vs TimeSeriesSplit Pearson train/test correlations and gaps.
    Useful for manual inspection of flagged features.
    """
    rows = []
    for feat in features:
        kf_train = results_kf[results_kf["feature"] == feat].set_index("fold")["pearson_train"]
        kf_test  = results_kf[results_kf["feature"] == feat].set_index("fold")["pearson_test"]
        ts_train = results_ts[results_ts["feature"] == feat].set_index("fold")["pearson_train"]
        ts_test  = results_ts[results_ts["feature"] == feat].set_index("fold")["pearson_test"]

        for fold in range(n_splits):
            rows.append({
                "feature":  feat,
                "fold":     fold,
                "kf_train": kf_train.get(fold),
                "kf_test":  kf_test.get(fold),
                "kf_gap":   kf_train.get(fold) - kf_test.get(fold),
                "ts_train": ts_train.get(fold),
                "ts_test":  ts_test.get(fold),
                "ts_gap":   ts_train.get(fold) - ts_test.get(fold),
            })
    return pd.DataFrame(rows)


def screen_non_iid_features(X, y, n_splits=5, ratio_threshold=2.0):
    """
    Screen for features that likely violate the IID assumption,
    using the variance of KFold vs TimeSeriesSplit correlation gaps.

    Core idea:
        - KFold ignores row order → gap variance stays low even for non-IID features
        - TimeSeriesSplit respects row order → gap variance inflates for non-IID features
        - A large ratio (TSSplit var / KFold var) signals a potential IID violation

    No time column or domain knowledge required.
    Intended as a first-pass screening tool; downstream validation is left to the user.

    Parameters:
        X               : feature DataFrame
        y               : target Series
        n_splits        : number of folds (default: 5)
        ratio_threshold : TSSplit/KFold variance ratio to flag a feature (default: 2.0)

    Returns:
        DataFrame: flagged features with pearson_ratio and spearman_ratio columns,
                   sorted by pearson_ratio descending.
    """
    kf  = KFold(n_splits=n_splits, shuffle=False)
    tss = TimeSeriesSplit(n_splits=n_splits)

    results_kf = compute_corr_gap(X, y, kf)
    results_ts = compute_corr_gap(X, y, tss)

    var_kf  = compute_gap_variance(results_kf, min_valid_folds=4).add_suffix("_kf").rename(columns={"feature_kf":  "feature"})
    var_tss = compute_gap_variance(results_ts, min_valid_folds=4).add_suffix("_tss").rename(columns={"feature_tss": "feature"})

    comparison = var_kf.merge(var_tss, on="feature")
    comparison["pearson_ratio"]  = comparison["pearson_gap_var_tss"]  / comparison["pearson_gap_var_kf"]
    comparison["spearman_ratio"] = comparison["spearman_gap_var_tss"] / comparison["spearman_gap_var_kf"]

    flagged = comparison[
        (comparison["pearson_ratio"]  > ratio_threshold) |
        (comparison["spearman_ratio"] > ratio_threshold)
    ][["feature", "pearson_ratio", "spearman_ratio"]].sort_values("pearson_ratio", ascending=False)

    return flagged

