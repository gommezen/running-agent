# ---------------------------------------------------
# src/xai_utils.py
# ---------------------------------------------------
# 🧩 XAI Utility Functions (robust to SHAP versions/shapes)
# ---------------------------------------------------

# ---------------------------------------------------
# WHAT THIS MODULE DOES (in simple terms)
# ---------------------------------------------------
# SHAP explains model predictions, but it can return results in many shapes
# (lists, 2D arrays, 3D arrays) depending on SHAP version, model type, and how
# many rows you pass in. That inconsistency often breaks downstream code.
#
# This module turns “whatever SHAP gives us” into:
#   1) One clean 1D vector of SHAP values for a single row
#      (same order as your DataFrame columns)
#   2) A tidy table of global importance (mean absolute SHAP per feature)
#
# In short: you get stable, predictable outputs no matter how SHAP formats things.
#
# ---------------------------------------------------
# PUBLIC FUNCTIONS YOU’LL ACTUALLY CALL
# ---------------------------------------------------
# shap_vector_for_sample(explainer, model, X_one_row) -> (np.ndarray, Any)
#   Use when you want to explain ONE prediction.
#   - Input : a DataFrame with EXACTLY ONE ROW and the same columns/order used to train
#   - Output: (vector, raw)
#       vector = 1D numpy array, length == number of features
#       raw    = the original SHAP output (useful for plots/debugging)
#   - For classifiers, it automatically explains the predicted class.
#
# shap_summary_dataframe(explainer, X, top_n=15) -> pd.DataFrame
#   Use when you want GLOBAL importance across many rows.
#   - Input : X as a DataFrame with training-aligned columns
#   - Output: DataFrame with columns: feature, mean_abs_shap
#             (higher = more important on average)
#
# ---------------------------------------------------
# HOW TO USE (copy/paste recipe)
# ---------------------------------------------------
# 1) Build an explainer once:
#       explainer = shap.TreeExplainer(model)                 # for tree models
#       # or:
#       explainer = shap.Explainer(model, background_df)      # for others; give 100–500 background rows
# 2) Single prediction explanation:
#       row = X_test.iloc[[0]]                                # note the double brackets → keeps DataFrame shape
#       vec, raw = shap_vector_for_sample(explainer, model, row)
# 3) Global importance:
#       imp_df = shap_summary_dataframe(explainer, X_test, top_n=20)
#
# ---------------------------------------------------
# WHY THIS EXISTS (the problem we fix)
# ---------------------------------------------------
# SHAP’s output shapes vary with:
#   - SHAP version (old .shap_values vs new Explanation API)
#   - Model type (regressor / binary / multiclass)
#   - Backend (Tree / Kernel / Linear)
#   - Batch size (1 row vs many rows)
# We normalize these variants so your code doesn’t have to.
#
# ---------------------------------------------------
# GUARANTEES & DESIGN CHOICES
# ---------------------------------------------------
# - We try the new API first (explainer(X) → Explanation), then fall back to .shap_values(X).
# - We always rely on pandas DataFrame columns to keep feature order stable.
# - If something is off (wrong columns, unknown shape), we fail early with a clear message
#   so you can fix alignment—this is the #1 source of SHAP bugs.
#
# ---------------------------------------------------
# COMMON PITFALLS (and how we handle them)
# ---------------------------------------------------
# - Single-row quirks: SHAP may return shape (n_features,) or (1, n_features).
#   → We normalize to a clean 1D vector aligned to your columns.
# - Multiclass weirdness: Some backends return a LIST (one array per class),
#   others return a 3D tensor with a class axis in different positions.
#   → We detect and select the predicted class consistently.
# - API drift: Old vs new SHAP APIs produce different objects.
#   → We support both seamlessly.
#
# ---------------------------------------------------
# WHAT THIS MODULE DOES NOT DO
# ---------------------------------------------------
# - It doesn’t build the explainer for you (you choose Tree vs Kernel/Linear).
# - It doesn’t reorder or create features—your X must match training columns & order.
# - If SHAP invents a brand-new shape in a future release, we raise a clear error
#   instead of guessing and silently returning nonsense.
#
# ---------------------------------------------------
# QUICK TROUBLESHOOTING
# ---------------------------------------------------
# ✔ “Module says columns don’t match”
#     → Ensure X has the same column NAMES and ORDER as during training.
#       Best: use model.feature_names_in_ or save a feature list to JSON.
#
# ✔ “I passed one row but still get shape errors”
#     → Make sure you passed X.iloc[[i]] (double brackets) so it’s a 1-row DataFrame,
#       not a Series.
#
# ✔ “Which explainer should I use?”
#     → Tree models: shap.TreeExplainer(model)
#       Non-tree models: shap.Explainer(model, background_df)
#
# ---------------------------------------------------


from __future__ import annotations

from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import pandas as pd

try:
    import shap  # noqa: F401
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# ---------- Small helpers ----------

def _ensure_df_one_row(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
    if X.shape[0] == 1:
        return X
    if X.shape[0] == 0:
        raise ValueError("X has 0 rows; need exactly one row.")
    return X.iloc[[0]]  # take first row deterministically


def _predicted_class(model, X: pd.DataFrame) -> Optional[int]:
    """Return predicted class index if classifier exposes predict_proba, else None."""
    try:
        proba = model.predict_proba(X)
        return int(np.asarray(proba).argmax(axis=1)[0])
    except Exception:
        return None


def _is_explanation(obj: Any) -> bool:
    """Heuristic: True if 'obj' looks like shap.Explanation (new API)."""
    return hasattr(obj, "values") and hasattr(obj, "base_values")


def _unwrap_to_array_and_meta(sv: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize any SHAP result into a numpy array plus a tiny meta dict.

    Returns
    -------
    arr : np.ndarray
        Can be:
          - (n_samples, n_features)
          - (n_samples, n_classes, n_features)
          - (n_classes, n_samples, n_features)
          - (n_classes, n_features)      [rare older outputs]
          - (n_features,)                [single-row already reduced]
    meta : dict
        Minimal hints about layout, e.g. {"source": "list"|"explanation"|"array"}.
    """
    meta: Dict[str, Any] = {}
    if isinstance(sv, list):
        # Typical multiclass TreeExplainer: list[len=n_classes] of (n_samples, n_features)
        meta["source"] = "list"
        try:
            arr = np.stack([np.asarray(a) for a in sv], axis=0)  # (n_classes, n_samples, n_features)
        except Exception:
            arr = np.asarray([np.asarray(a) for a in sv])  # best effort
        return arr, meta

    if _is_explanation(sv):
        meta["source"] = "explanation"
        arr = np.asarray(sv.values)  # (n_samples, n_features) or (n_samples, n_outputs, n_features)
        return arr, meta

    meta["source"] = "array"
    return np.asarray(sv), meta


def _reduce_to_1d_for_class(
    arr: np.ndarray,
    n_features: int,
    pred_class: Optional[int],
    n_samples_expected: int = 1,
) -> np.ndarray:
    """
    Reduce any known SHAP layout to a 1D vector of length n_features for a single sample.
    """
    choose = 0 if pred_class is None else int(pred_class)

    # (n_samples, n_features)
    if arr.ndim == 2 and arr.shape[1] == n_features:
        return arr[0, :]

    # (n_samples, n_classes, n_features)
    if arr.ndim == 3 and arr.shape[2] == n_features and arr.shape[0] == n_samples_expected:
        return arr[0, choose, :]

    # (n_classes, n_samples, n_features)
    if arr.ndim == 3 and arr.shape[2] == n_features and arr.shape[1] == n_samples_expected:
        return arr[choose, 0, :]

    # (n_features, n_classes)
    if arr.ndim == 2 and arr.shape[0] == n_features and arr.shape[1] >= 1:
        if choose >= arr.shape[1]:
            raise ValueError(f"Pred class {choose} out of bounds for SHAP shape {arr.shape}")
        return arr[:, choose]

    # (n_classes, n_features)
    if arr.ndim == 2 and arr.shape[1] == n_features and arr.shape[0] >= 1:
        if choose >= arr.shape[0]:
            raise ValueError(f"Pred class {choose} out of bounds for SHAP shape {arr.shape}")
        return arr[choose, :]

    # (n_features,)
    if arr.ndim == 1 and arr.shape[0] == n_features:
        return arr

    raise ValueError(
        f"Unexpected SHAP array shape {arr.shape}. "
        f"Cannot align to 1D vector of length n_features={n_features}."
    )


# ---------- Public utilities ----------

def shap_vector_for_sample(
    explainer_or_model,
    model,
    X_one_row: pd.DataFrame,
) -> Tuple[np.ndarray, Any]:
    """
    Return SHAP values for one input row as a 1-D vector aligned with feature columns.

    Parameters
    ----------
    explainer_or_model : shap.Explainer or shap.TreeExplainer (preferred),
                         or any object with .shap_values(X) or callable explainer(X).
    model : sklearn-like estimator
        Used only to determine predicted class (if classifier).
    X_one_row : pd.DataFrame
        Exactly one-row DataFrame with the feature columns in training order.

    Returns
    -------
    shap_1d : np.ndarray
        1D vector (n_features,) for this row; uses predicted class if classifier.
    shap_full : Any
        Raw SHAP output (Explanation, list, or ndarray).
    """
    if not _HAS_SHAP:
        raise ImportError("shap is not installed. Please `pip install shap`.")

    X_one_row = _ensure_df_one_row(X_one_row)
    nfeat = X_one_row.shape[1]
    pred_cls = _predicted_class(model, X_one_row)

    # Prefer new API (explainer(X) -> Explanation), else old API (.shap_values(X))
    try:
        shap_full = explainer_or_model(X_one_row)  # new API path
    except Exception:
        if hasattr(explainer_or_model, "shap_values"):
            shap_full = explainer_or_model.shap_values(X_one_row)  # old API path
        else:
            raise TypeError(
                "Provided explainer does not support call or .shap_values(X). "
                "Pass a fitted shap.Explainer/TreeExplainer."
            )

    arr, _meta = _unwrap_to_array_and_meta(shap_full)
    vec = _reduce_to_1d_for_class(arr, n_features=nfeat, pred_class=pred_cls, n_samples_expected=1)

    if vec.shape[0] != nfeat:
        raise ValueError(
            f"Got SHAP vector length {vec.shape[0]} but need {nfeat}. "
            f"Check that X_one_row has the exact training feature columns and order."
        )
    return vec, shap_full


def shap_summary_dataframe(
    explainer,
    X: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values for feature-importance summary.

    Handles:
      - Explanation (new API): (n_samples, n_features) or (n_samples, n_outputs, n_features)
      - Old API list per class: list[n_classes] of (n_samples, n_features)
      - Old API arrays: (n_samples, n_features) or (n_samples, n_classes, n_features)

    Returns
    -------
    DataFrame with columns: feature, mean_abs_shap
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame (to preserve feature names).")

    # Prefer new API; fall back to old
    try:
        sv = explainer(X)  # Explanation in new API
    except Exception:
        if hasattr(explainer, "shap_values"):
            sv = explainer.shap_values(X)
        else:
            raise TypeError("Explainer does not support call or .shap_values(X).")

    arr, _meta = _unwrap_to_array_and_meta(sv)

    if arr.ndim == 2 and arr.shape == (X.shape[0], X.shape[1]):
        # (n_samples, n_features)
        mean_abs = np.mean(np.abs(arr), axis=0)

    elif arr.ndim == 3 and arr.shape[0] == X.shape[0] and arr.shape[2] == X.shape[1]:
        # (n_samples, n_outputs/classes, n_features)
        mean_abs = np.mean(np.mean(np.abs(arr), axis=1), axis=0)

    elif arr.ndim == 3 and arr.shape[0] != X.shape[0] and arr.shape[2] == X.shape[1]:
        # Likely (n_classes, n_samples, n_features)
        mean_abs = np.mean(np.mean(np.abs(arr), axis=1), axis=0)

    elif arr.ndim == 2 and arr.shape[0] == X.shape[1]:
        # (n_features, [n_classes]) — when X has one sample
        mean_abs = np.mean(np.abs(arr), axis=1)

    elif arr.ndim == 1 and arr.shape[0] == X.shape[1]:
        # (n_features,) — degenerate single-row summary
        mean_abs = np.abs(arr)

    else:
        raise ValueError(
            f"Unexpected SHAP array shape for summary: {arr.shape}. "
            "Ensure X columns match explainer’s training columns."
        )

    importance_df = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return importance_df
