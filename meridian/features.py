"""Feature engineering: realized volatility estimators, HAR features, targets.

All features at row t use only information available at the close of day t.
The target is next-day realized variance (t+1), so nothing here peeks ahead
except the explicit target column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EPS = 1e-12


def realized_variance(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Daily realized-variance proxies from OHLC (no intraday needed).

    Returns a frame with:
      ret       : close-to-close log return
      rv_cc     : squared log return (noisy 1-obs estimator)
      rv_park   : Parkinson (high-low range) variance estimator
      rv_gk     : Garman-Klass estimator (uses OHLC)
      rv        : primary target basis = Garman-Klass, floored
    """
    o, h, l, c = (np.log(ohlc[x]) for x in ("open", "high", "low", "close"))
    ret = c.diff()

    park = (h - l) ** 2 / (4.0 * np.log(2.0))
    gk = 0.5 * (h - l) ** 2 - (2.0 * np.log(2.0) - 1.0) * (c - o) ** 2

    out = pd.DataFrame(index=ohlc.index)
    out["ret"] = ret
    out["rv_cc"] = ret ** 2
    out["rv_park"] = park.clip(lower=EPS)
    out["rv_gk"] = gk.clip(lower=EPS)
    out["rv"] = out["rv_gk"].clip(lower=EPS)
    return out


def har_features(rv: pd.Series) -> pd.DataFrame:
    """Corsi (2009) HAR components on realized variance.

    rv_d = RV_t, rv_w = mean RV over last 5d, rv_m = mean RV over last 22d.
    All are known at close of day t.
    """
    f = pd.DataFrame(index=rv.index)
    f["rv_d"] = rv
    f["rv_w"] = rv.rolling(5).mean()
    f["rv_m"] = rv.rolling(22).mean()
    return f


def semivariance_features(ret: pd.Series) -> pd.DataFrame:
    """Realized semivariance HAR terms (Patton-Sheppard): good/bad decomposition
    of close-to-close squared returns. Bad (downside) semivariance carries more
    predictive information for next-day variance than the symmetric HAR terms
    alone (validated in scripts/run_semivar_ml.py). All at t, no leakage.

    nsv_d/w/m = log bad (downside) semivariance at day/week(5d)/month(22d).
    psv_d     = log good (upside) semivariance at day.
    """
    nsv = ret.clip(upper=0) ** 2
    psv = ret.clip(lower=0) ** 2
    f = pd.DataFrame(index=ret.index)
    f["nsv_d"] = np.log(nsv + EPS)
    f["nsv_w"] = np.log(nsv.rolling(5).mean() + EPS)
    f["nsv_m"] = np.log(nsv.rolling(22).mean() + EPS)
    f["psv_d"] = np.log(psv + EPS)
    return f


def build_asset_frame(ohlc: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full per-asset modeling frame with features (t) and targets (t+1, t+5, t+22).

    Columns:
      log-space HAR features, semivariance features, realized measures, optional
      macro context, and `y`/`y_5`/`y_22` = mean log RV over the next 1/5/22 days
      as the multi-horizon prediction targets.
    """
    rvf = realized_variance(ohlc)
    rv = rvf["rv"]

    df = pd.DataFrame(index=ohlc.index)
    df["ret"] = rvf["ret"]
    df["rv"] = rv
    df["log_rv"] = np.log(rv + EPS)

    har = har_features(rv)
    df["har_d"] = np.log(har["rv_d"] + EPS)
    df["har_w"] = np.log(har["rv_w"] + EPS)
    df["har_m"] = np.log(har["rv_m"] + EPS)

    # extra descriptive context features (available at t)
    df["ret_abs"] = rvf["ret"].abs()
    df["ret_5"] = rvf["ret"].rolling(5).sum()
    df["rv_cc"] = np.log(rvf["rv_cc"].clip(lower=EPS) + EPS)

    sv = semivariance_features(rvf["ret"])
    df["nsv_d"] = sv["nsv_d"]
    df["nsv_w"] = sv["nsv_w"]
    df["nsv_m"] = sv["nsv_m"]
    df["psv_d"] = sv["psv_d"]

    if macro is not None:
        m = macro.reindex(df.index).ffill()
        df["vix"] = np.log(m["VIXCLS"].clip(lower=EPS))
        df["term"] = (m["DGS10"] - m["DGS2"])  # yield-curve slope
        df["vix_chg"] = df["vix"].diff()

    # TARGETS: forward mean log realized variance at h=1, 5, 22 days.
    # h=1 uses the plain next-day value; h=5/22 average *future* RV (days
    # t+1..t+h, never including t) so all three targets are directly comparable
    # QLIKE targets at their own horizon, not lookahead-tainted rolling means.
    df["y"] = df["log_rv"].shift(-1)
    fwd_rv = rv.shift(-1)
    df["y_5"] = np.log(fwd_rv.rolling(5).mean().shift(-4) + EPS)
    df["y_22"] = np.log(fwd_rv.rolling(22).mean().shift(-21) + EPS)
    # raw (non-log) forward RV truth, for QLIKE evaluation at each horizon
    df["rv_next"] = fwd_rv
    df["rv_next_5"] = fwd_rv.rolling(5).mean().shift(-4)
    df["rv_next_22"] = fwd_rv.rolling(22).mean().shift(-21)
    # secondary target: next-day close-to-close return (for the distributional/VaR head)
    df["r_next"] = rvf["ret"].shift(-1)

    return df


if __name__ == "__main__":
    from meridian.data import load_all

    d = load_all()
    spy = build_asset_frame(d["prices"]["SPY"], d["macro"])
    print(spy.tail(6).round(4).to_string())
    print("\nshape:", spy.shape, "| NaNs per col:")
    print(spy.isna().sum().to_string())
