# ---------------------------------------------------
# src/xai_utils.py
# ---------------------------------------------------
# 🧩 XAI Utility Functions (robust to SHAP versions/shapes)
# ---------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# ---------- Small helpers ----------


def _ensure_df_one_row(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure input is a DataFrame with exactly one row."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
    if X.shape[0] == 1:
        return X
    if X.shape[0] == 0:
        raise ValueError("X has 0 rows; need exactly one row.")
    return X.iloc[[0]]


def _predicted_class(model: Any, X: pd.DataFrame) -> int | None:
    """Return predicted class index if classifier exposes predict_proba, else None."""
    try:
        proba = model.predict_proba(X)
        return int(np.asarray(proba).argmax(axis=1)[0])
    except Exception:
        return None


def _is_explanation(obj: Any) -> bool:
    """Heuristic: True if 'obj' looks like shap.Explanation (new API)."""
    return hasattr(obj, "values") and hasattr(obj, "base_values")


def _unwrap_to_array_and_meta(sv: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Normalize any SHAP result into a numpy array plus a tiny meta dict.
    """
    meta: dict[str, Any] = {}

    if isinstance(sv, list):
        meta["source"] = "list"
        try:
            arr = np.stack([np.asarray(a) for a in sv], axis=0)
        except Exception:
            arr = np.asarray([np.asarray(a) for a in sv])
        return arr, meta

    if _is_explanation(sv):
        meta["source"] = "explanation"
        arr = np.asarray(sv.values)
        return arr, meta

    meta["source"] = "array"
    return np.asarray(sv), meta


def _reduce_to_1d_for_class(
    arr: np.ndarray,
    n_features: int,
    pred_class: int | None,
    n_samples_expected: int = 1,
) -> np.ndarray:
    """Reduce any known SHAP layout to a 1D vector for a single sample."""
    choose = 0 if pred_class is None else int(pred_class)

    if arr.ndim == 2 and arr.shape[1] == n_features:
        return arr[0, :]

    # Shape (n_samples, n_features, n_classes) — pick class from last axis
    if arr.ndim == 3 and arr.shape[1] == n_features and arr.shape[0] == n_samples_expected:
        return arr[0, :, choose]

    # Shape (n_samples, n_classes, n_features) — pick class from middle axis
    if arr.ndim == 3 and arr.shape[2] == n_features and arr.shape[0] == n_samples_expected:
        return arr[0, choose, :]

    if arr.ndim == 3 and arr.shape[2] == n_features and arr.shape[1] == n_samples_expected:
        return arr[choose, 0, :]

    if arr.ndim == 2 and arr.shape[0] == n_features and arr.shape[1] >= 1:
        if choose >= arr.shape[1]:
            raise ValueError(f"Pred class {choose} out of bounds for SHAP shape {arr.shape}")
        return arr[:, choose]

    if arr.ndim == 2 and arr.shape[1] == n_features and arr.shape[0] >= 1:
        if choose >= arr.shape[0]:
            raise ValueError(f"Pred class {choose} out of bounds for SHAP shape {arr.shape}")
        return arr[choose, :]

    if arr.ndim == 1 and arr.shape[0] == n_features:
        return arr

    raise ValueError(
        f"Unexpected SHAP array shape {arr.shape}. "
        f"Cannot align to 1D vector of length n_features={n_features}."
    )


# ---------- Public utilities ----------


def shap_vector_for_sample(
    explainer_or_model: Any,
    model: Any,
    X_one_row: pd.DataFrame,
) -> tuple[np.ndarray, Any]:
    """Return SHAP values for one input row as a 1-D vector aligned with feature columns."""
    if not _HAS_SHAP:
        raise ImportError("shap is not installed. Please `pip install shap`.")

    X_one_row = _ensure_df_one_row(X_one_row)
    nfeat = X_one_row.shape[1]
    pred_cls = _predicted_class(model, X_one_row)

    try:
        shap_full = explainer_or_model(X_one_row)  # new API
    except Exception:
        if hasattr(explainer_or_model, "shap_values"):
            shap_full = explainer_or_model.shap_values(X_one_row)
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
            "Check that X_one_row has the exact training feature columns and order."
        )
    return vec, shap_full


def shap_summary_dataframe(
    explainer: Any,
    X: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """Compute mean absolute SHAP values for feature-importance summary."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame (to preserve feature names).")

    try:
        sv = explainer(X)
    except Exception:
        if hasattr(explainer, "shap_values"):
            sv = explainer.shap_values(X)
        else:
            raise TypeError("Explainer does not support call or .shap_values(X).")

    arr, _meta = _unwrap_to_array_and_meta(sv)

    if arr.ndim == 2 and arr.shape == (X.shape[0], X.shape[1]):
        mean_abs = np.mean(np.abs(arr), axis=0)
    elif arr.ndim == 3 and arr.shape[0] == X.shape[0] and arr.shape[1] == X.shape[1]:
        # Shape (n_samples, n_features, n_classes) — mean over samples & classes
        mean_abs = np.mean(np.abs(arr), axis=(0, 2))
    elif (
        arr.ndim == 3
        and arr.shape[0] == X.shape[0]
        and arr.shape[2] == X.shape[1]
        or arr.ndim == 3
        and arr.shape[0] != X.shape[0]
        and arr.shape[2] == X.shape[1]
    ):
        # Shape (n_samples, n_classes, n_features) — mean over samples & classes
        mean_abs = np.mean(np.mean(np.abs(arr), axis=1), axis=0)
    elif arr.ndim == 2 and arr.shape[0] == X.shape[1]:
        mean_abs = np.mean(np.abs(arr), axis=1)
    elif arr.ndim == 1 and arr.shape[0] == X.shape[1]:
        mean_abs = np.abs(arr)
    else:
        raise ValueError(
            f"Unexpected SHAP array shape for summary: {arr.shape}. "
            "Ensure X columns match explainer’s training columns."
        )

    return (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
