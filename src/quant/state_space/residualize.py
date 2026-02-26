from __future__ import annotations

import numpy as np
import pandas as pd


def ols_residual(y: pd.Series, *regressors: pd.Series) -> pd.Series:
    if len(regressors) == 0:
        raise ValueError("ols_residual requires at least one regressor")

    idx = y.index
    yv = y.to_numpy(dtype=float)
    x_cols = [r.reindex(idx).to_numpy(dtype=float) for r in regressors]
    X = np.column_stack(x_cols)

    valid = np.isfinite(yv) & np.isfinite(X).all(axis=1)
    resid = np.full_like(yv, fill_value=np.nan, dtype=float)
    if valid.sum() < (X.shape[1] + 1):
        return pd.Series(resid, index=idx, name=f"{y.name}_res")

    Xv = X[valid]
    y_valid = yv[valid]
    X_design = np.column_stack([np.ones(len(Xv), dtype=float), Xv])
    beta, _, _, _ = np.linalg.lstsq(X_design, y_valid, rcond=None)
    y_hat = X_design @ beta
    resid[valid] = y_valid - y_hat
    return pd.Series(resid, index=idx, name=f"{y.name}_res")


def residualize_axes(X_raw: pd.Series, Y_raw: pd.Series, Z_raw: pd.Series) -> pd.DataFrame:
    Y_res = ols_residual(Y_raw, X_raw).rename("Y_res")
    Z_res = ols_residual(Z_raw, X_raw, Y_res).rename("Z_res")
    return pd.DataFrame({"Y_res": Y_res, "Z_res": Z_res}, index=X_raw.index)
