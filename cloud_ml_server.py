import os
import time
import math
import io
import base64
import threading
import traceback
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

warnings.filterwarnings("ignore")

# Firebase REST layer
import firebase_db as _db

_env_url = os.getenv("FIREBASE_URL", "").strip()
if _env_url:
    _db.FIREBASE_URL = _env_url
    print(f"[CONFIG] Firebase URL from env: {_db.FIREBASE_URL}")
else:
    print(f"[CONFIG] Firebase URL from module: {_db.FIREBASE_URL}")

db = _db

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
RETRAIN_EVERY = 20
MIN_ROWS = 50
ROLLING_WINDOW = 1000
TRAIN_INTERVAL = 120
STARTUP_TRAIN_DELAY = 5
FEED_SCHEDULE_HOURS = [3, 8, 11, 14]

# Minimum meaningful changes. These reduce noise from load cell vibration and
# very small flow pulses while still allowing real poultry consumption to count.
MIN_FEED_DROP_KG = 0.01       # 10 g feed change
MIN_WATER_DELTA_L = 0.001     # 1 mL water change
MIN_VALID_DAYS_FOR_MODEL = 5
MIN_VALID_DAYS_FOR_BASELINE = 1

# Safety caps. Change only if your actual system uses bigger containers.
MAX_REASONABLE_FEED_KG = 100.0
MAX_REASONABLE_WATER_L = 10000.0
ANOMALY_Z = 2.5

BG = "#0e1117"
BORDER = "#3d4257"
MUTED = "#a0aec0"
CFEED = "#50C8FF"
CWATER = "#1f77b4"

DAILY_FEATURES = [
    "day_index", "day_of_week", "month",
    "records", "avg_flow", "flow_active_rate", "level_ok_rate",
    "mean_feed_remaining", "latest_water_total",
    "feed_prev_day", "water_prev_day",
    "roll3_feed_target", "roll3_water_target",
]

state: Dict[str, Any] = {
    "trained_rows": 0,
    "valid_rows": 0,
    "valid_days": 0,
    "removed_rows": 0,
    "training": False,
    "status": "starting",
    "error": "",
    "model_mode": "starting",
}

_lock = threading.Lock()
_event = threading.Event()
_workers_started = False


# ═════════════════════════════════════════════════════════════════════════════
# BASIC HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except Exception:
        return lo


def _parse_date(rec: Dict[str, Any]) -> pd.Timestamp:
    """Parse ESP timestamp. Returns NaT if no usable timestamp exists."""
    ts_val = rec.get("timestamp")
    try:
        ts_num = float(ts_val)
        if ts_num > 1_000_000:
            return pd.to_datetime(ts_num, unit="s", errors="coerce")
    except Exception:
        pass

    for key in ["ts", "date", "datetime", "createdAt"]:
        text = str(rec.get(key, "") or "").strip()
        if not text:
            continue
        try:
            return pd.to_datetime(text, errors="coerce")
        except Exception:
            pass
    return pd.NaT


def _first_number(rec: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for key in keys:
        if key in rec:
            return _safe_float(rec.get(key), default)
    return default


def _level_to_ok(value) -> int:
    """Convert water-level text into 1=available, 0=low/empty."""
    t = str(value or "").strip().lower()
    if not t:
        return 0
    low_words = ["empty", "low", "0%", "no water", "dry", "false", "off", "fill"]
    if any(w in t for w in low_words):
        return 0
    return 1


def _status_has_error(rec: Dict[str, Any]) -> bool:
    """Detect known sensor-error messages without treating flow=0 as error."""
    status_keys = [
        "status", "sensorStatus", "loadStatus", "loadCellStatus", "hx711Status",
        "flowStatus", "waterStatus", "error", "message",
    ]
    joined = " ".join(str(rec.get(k, "") or "") for k in status_keys).strip().lower()
    if not joined:
        return False
    bad_words = [
        "not found", "missing", "disconnected", "offline", "nan", "invalid",
        "sensor error", "hx711 error", "loadcell error", "failed", "fail",
    ]
    return any(w in joined for w in bad_words)


def _schedule_label(dt: pd.Timestamp) -> str:
    try:
        return dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return str(dt)


def _robust_recent_average(series: pd.Series, minimum: float = 0.0, tail: int = 3) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    vals = vals[vals > minimum]
    if vals.empty:
        return 0.0
    vals = vals.tail(tail)
    # Median is safer than plain mean when one day has a sensor spike.
    return float(vals.median())


def _round2(x: float) -> float:
    return round(max(0.0, float(x or 0.0)), 2)


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLEANING + CONSUMPTION CALCULATION
# ═════════════════════════════════════════════════════════════════════════════

def readings_to_professional_df(readings: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert raw Firebase rows into clean sensor dataframe.

    This intentionally does NOT remove rows where flow is 0.00, because that is
    normal when chickens are not drinking.
    """
    rows = []
    raw_rows = len(readings or [])
    skipped_bad_timestamp = 0
    skipped_error_status = 0
    skipped_range = 0

    for rec in readings or []:
        if not isinstance(rec, dict):
            continue

        date = _parse_date(rec)
        if pd.isna(date):
            skipped_bad_timestamp += 1
            continue

        feed_kg = _first_number(rec, ["weight", "feed_kg", "feedKg", "feed", "currentWeight"], 0.0)
        water_liters = _first_number(rec, ["totalLiters", "water_liters", "waterL", "total_liters", "waterTotal"], 0.0)
        flow = _first_number(rec, ["flow", "flowRate", "waterFlow", "flow_lpm"], 0.0)

        if feed_kg < 0 or water_liters < 0 or flow < 0:
            skipped_range += 1
            continue
        if feed_kg > MAX_REASONABLE_FEED_KG or water_liters > MAX_REASONABLE_WATER_L:
            skipped_range += 1
            continue

        error_flag = _status_has_error(rec)
        if error_flag:
            skipped_error_status += 1
            continue

        level = str(rec.get("level", rec.get("waterLevel", rec.get("levelStatus", ""))) or "")
        day_of_week = _safe_int(rec.get("dayOfWeek"), pd.Timestamp(date).dayofweek)
        month = _safe_int(rec.get("month"), pd.Timestamp(date).month)

        rows.append({
            "date": date,
            "feed_kg": feed_kg,
            "water_liters": water_liters,
            "flow": flow,
            "level": level,
            "level_ok": _level_to_ok(level),
            "day_of_week": day_of_week,
            "month": month,
            "system": _safe_int(rec.get("system"), 1),
        })

    if not rows:
        stats = {
            "rawRows": raw_rows,
            "validRows": 0,
            "removedRows": raw_rows,
            "skippedBadTimestamp": skipped_bad_timestamp,
            "skippedSensorError": skipped_error_status,
            "skippedOutOfRange": skipped_range,
            "note": "No valid timestamped sensor rows available.",
        }
        return pd.DataFrame(), stats

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    try:
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        pass

    # Soft caps for sudden spikes. This does not force values to zero; it only
    # prevents one abnormal row from dominating model training.
    for col in ["feed_kg", "water_liters", "flow"]:
        vals = df[col][df[col] > 0]
        if len(vals) >= 20:
            q99 = float(vals.quantile(0.99))
            if q99 > 0:
                df[col] = df[col].clip(upper=q99 * 1.50)

    stats = {
        "rawRows": raw_rows,
        "validRows": int(len(df)),
        "removedRows": int(max(0, raw_rows - len(df))),
        "skippedBadTimestamp": int(skipped_bad_timestamp),
        "skippedSensorError": int(skipped_error_status),
        "skippedOutOfRange": int(skipped_range),
        "flowZeroIsNormal": True,
    }
    return df, stats


def add_consumption_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute actual feed/water consumed from changes between readings."""
    if df.empty:
        return df

    out = df.copy().sort_values("date").reset_index(drop=True)
    out["hour"] = out["date"].dt.hour
    out["date_day"] = out["date"].dt.date
    out["flow_active"] = (out["flow"] > 0).astype(int)

    # Feed: consumption is a DROP in remaining feed weight.
    feed_drop = out["feed_kg"].shift(1) - out["feed_kg"]
    feed_drop = feed_drop.fillna(0.0)
    feed_drop = feed_drop.where(feed_drop >= MIN_FEED_DROP_KG, 0.0)

    # Water: consumption is an INCREASE in accumulated total liters.
    water_delta = out["water_liters"] - out["water_liters"].shift(1)
    water_delta = water_delta.fillna(0.0)
    water_delta = water_delta.where(water_delta >= MIN_WATER_DELTA_L, 0.0)

    # Robustly cap impossible one-row jumps. This protects the ML from sensor
    # resets/disconnections while still preserving normal consumption data.
    for name, series in [("feed_consumed", feed_drop), ("water_consumed", water_delta)]:
        positive = series[series > 0]
        if len(positive) >= 20:
            median = float(positive.median())
            q95 = float(positive.quantile(0.95))
            cap = max(q95 * 2.0, median * 8.0, 0.01)
            series = series.clip(upper=cap)
        out[name] = series.clip(lower=0.0)

    return out


def build_daily_consumption_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate readings into daily feed/water consumption targets."""
    if df.empty:
        return pd.DataFrame()

    g = df.groupby("date_day", sort=True)
    daily = g.agg(
        date=("date", "max"),
        feed_target_kg=("feed_consumed", "sum"),
        water_target_l=("water_consumed", "sum"),
        records=("date", "count"),
        avg_flow=("flow", "mean"),
        flow_active_rate=("flow_active", "mean"),
        level_ok_rate=("level_ok", "mean"),
        mean_feed_remaining=("feed_kg", "mean"),
        latest_water_total=("water_liters", "last"),
    ).reset_index(drop=True)

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    daily["day_index"] = np.arange(len(daily), dtype=float)
    daily["day_of_week"] = daily["date"].dt.dayofweek.astype(int)
    daily["month"] = daily["date"].dt.month.astype(int)

    daily["feed_prev_day"] = daily["feed_target_kg"].shift(1).fillna(0.0)
    daily["water_prev_day"] = daily["water_target_l"].shift(1).fillna(0.0)
    daily["roll3_feed_target"] = daily["feed_target_kg"].rolling(3, min_periods=1).median()
    daily["roll3_water_target"] = daily["water_target_l"].rolling(3, min_periods=1).median()

    for f in DAILY_FEATURES:
        if f not in daily:
            daily[f] = 0.0
        daily[f] = pd.to_numeric(daily[f], errors="coerce").fillna(0.0)

    daily["valid_feed_day"] = daily["feed_target_kg"] > MIN_FEED_DROP_KG
    daily["valid_water_day"] = daily["water_target_l"] > MIN_WATER_DELTA_L
    daily["valid_any_day"] = daily["valid_feed_day"] | daily["valid_water_day"]
    return daily


# ═════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL ML TRAINING / METRICS
# ═════════════════════════════════════════════════════════════════════════════

def make_model():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.045,
            max_depth=3,
            subsample=0.85,
            random_state=42,
        )),
    ])


def _calc_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "score": 0.50}
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    mean_target = float(np.mean(np.abs(y_true)))
    score = _clamp(1.0 - mae / (mean_target + 1e-6), 0.0, 1.0) if mean_target > 0 else 0.50
    return {"mae": mae, "rmse": rmse, "mape": mape, "score": score}


def train_target_model(daily: pd.DataFrame, target_col: str, valid_col: str) -> Dict[str, Any]:
    """Train one daily target model. Falls back to baseline if days are few."""
    valid = daily[daily[valid_col]].copy()
    valid = valid[valid[target_col] > 0].reset_index(drop=True)

    baseline = _robust_recent_average(valid[target_col], minimum=0.0, tail=3)
    out = {
        "model": None,
        "mode": "baseline",
        "baseline": baseline,
        "valid_days": int(len(valid)),
        "metrics": {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "score": 0.50},
    }

    if len(valid) < MIN_VALID_DAYS_FOR_MODEL or valid[target_col].nunique() < 2:
        return out

    X = valid[DAILY_FEATURES].fillna(0.0)
    y = valid[target_col].astype(float)

    model = make_model()

    # Time-series holdout: older rows train, newest rows validate.
    test_n = max(1, min(3, int(round(len(valid) * 0.25))))
    if len(valid) - test_n >= 3:
        x_train, x_test = X.iloc[:-test_n], X.iloc[-test_n:]
        y_train, y_test = y.iloc[:-test_n], y.iloc[-test_n:]
        trial = make_model()
        trial.fit(x_train, y_train)
        pred = np.maximum(0.0, trial.predict(x_test))
        metrics = _calc_metrics(y_test.values, pred)
    else:
        metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "score": 0.55}

    model.fit(X, y)
    out.update({"model": model, "mode": "ml", "metrics": metrics})
    return out


def build_next_daily_features(daily: pd.DataFrame, next_date: pd.Timestamp) -> pd.DataFrame:
    if daily.empty:
        row = {f: 0.0 for f in DAILY_FEATURES}
        row.update({"day_of_week": next_date.dayofweek, "month": next_date.month})
        return pd.DataFrame([row])[DAILY_FEATURES]

    last = daily.iloc[-1]
    row = {
        "day_index": float(last.get("day_index", len(daily) - 1)) + 1.0,
        "day_of_week": int(next_date.dayofweek),
        "month": int(next_date.month),
        "records": float(max(1.0, daily["records"].tail(3).median())),
        "avg_flow": float(max(0.0, daily["avg_flow"].tail(3).median())),
        "flow_active_rate": float(_clamp(daily["flow_active_rate"].tail(3).median(), 0.0, 1.0)),
        "level_ok_rate": float(_clamp(daily["level_ok_rate"].tail(3).median(), 0.0, 1.0)),
        "mean_feed_remaining": float(max(0.0, daily["mean_feed_remaining"].tail(3).median())),
        "latest_water_total": float(max(0.0, daily["latest_water_total"].tail(1).iloc[-1])),
        "feed_prev_day": float(max(0.0, daily["feed_target_kg"].tail(1).iloc[-1])),
        "water_prev_day": float(max(0.0, daily["water_target_l"].tail(1).iloc[-1])),
        "roll3_feed_target": float(max(0.0, daily["feed_target_kg"].tail(3).median())),
        "roll3_water_target": float(max(0.0, daily["water_target_l"].tail(3).median())),
    }
    return pd.DataFrame([{k: row.get(k, 0.0) for k in DAILY_FEATURES}])


def predict_target(model_info: Dict[str, Any], daily: pd.DataFrame, target_col: str,
                   next_features: pd.DataFrame, day_of_week: int) -> Tuple[float, str]:
    """Predict one target and blend with recent/seasonal baseline for stability."""
    recent = _robust_recent_average(daily[target_col], minimum=0.0, tail=3)

    seasonal = 0.0
    try:
        same_dow = daily[(daily["day_of_week"] == day_of_week) & (daily[target_col] > 0)]
        seasonal = _robust_recent_average(same_dow[target_col], minimum=0.0, tail=4)
    except Exception:
        pass

    baseline = recent or model_info.get("baseline", 0.0) or seasonal
    if model_info.get("model") is not None:
        try:
            model_v = max(0.0, float(model_info["model"].predict(next_features)[0]))
            # If the model outputs zero while real recent consumption exists, do not
            # allow false zero. Blend with real consumption baseline.
            if baseline > 0 and model_v < baseline * 0.10:
                final = baseline
                source = "ml_corrected_by_recent_consumption"
            else:
                parts = [model_v * 0.60]
                weight = 0.60
                if recent > 0:
                    parts.append(recent * 0.30)
                    weight += 0.30
                if seasonal > 0:
                    parts.append(seasonal * 0.10)
                    weight += 0.10
                final = sum(parts) / max(weight, 1e-6)
                source = "ml_consumption_blend"
            return _round2(final), source
        except Exception as e:
            print(f"[ML] prediction fallback for {target_col}: {e}")

    return _round2(baseline), "baseline_recent_consumption"


def arima_forecast(series: pd.Series, fallback: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    vals = vals[vals > 0]
    if len(vals) < 7 or vals.nunique() < 2:
        return _round2(fallback)
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(vals.values, order=(1, 1, 1), enforce_stationarity=False,
                      enforce_invertibility=False).fit()
        return _round2(float(model.forecast(1)[0]))
    except Exception as e:
        print(f"[ML] ARIMA skipped: {e}")
        return _round2(fallback)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIDENCE, TREND, ANOMALY, FORECAST, CHART
# ═════════════════════════════════════════════════════════════════════════════

def confidence_label(c: float) -> str:
    if c >= 0.80:
        return "High"
    if c >= 0.55:
        return "Medium"
    if c >= 0.30:
        return "Low"
    return "Very Low"


def calc_confidence(raw_stats: Dict[str, Any], daily: pd.DataFrame,
                    feed_info: Dict[str, Any], water_info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    raw_rows = int(raw_stats.get("rawRows", 0) or 0)
    valid_rows = int(raw_stats.get("validRows", 0) or 0)
    removed_rows = int(raw_stats.get("removedRows", 0) or 0)
    valid_days = int(daily["valid_any_day"].sum()) if not daily.empty else 0

    row_score = _clamp((valid_rows - MIN_ROWS) / max(1, 250 - MIN_ROWS), 0.0, 1.0)
    day_score = _clamp(valid_days / 14.0, 0.0, 1.0)
    quality_score = _clamp(valid_rows / max(1, raw_rows), 0.0, 1.0)
    removal_penalty = _clamp(1.0 - removed_rows / max(1, raw_rows), 0.0, 1.0)

    try:
        latest_dt = pd.to_datetime(daily["date"].max()) if not daily.empty else pd.NaT
        age_hours = max(0.0, (pd.Timestamp.now() - latest_dt).total_seconds() / 3600.0)
        recency_score = _clamp(1.0 - max(0.0, age_hours - 2.0) / 72.0, 0.0, 1.0)
    except Exception:
        age_hours = None
        recency_score = 0.50

    feed_score = float(feed_info.get("metrics", {}).get("score", 0.50))
    water_score = float(water_info.get("metrics", {}).get("score", 0.50))
    model_score = _clamp((feed_score + water_score) / 2.0, 0.0, 1.0)

    activity_score = 0.0
    if not daily.empty:
        activity_score = _clamp(float(daily["valid_any_day"].mean()) * 1.50, 0.0, 1.0)

    # Professional confidence is multi-factor, not just row count.
    conf = (
        row_score * 0.15 +
        day_score * 0.25 +
        quality_score * 0.15 +
        removal_penalty * 0.10 +
        recency_score * 0.10 +
        model_score * 0.15 +
        activity_score * 0.10
    )
    conf = round(_clamp(conf, 0.0, 1.0), 2)

    details = {
        "rowScore": round(row_score, 2),
        "dayScore": round(day_score, 2),
        "dataQualityScore": round(quality_score, 2),
        "removedRowsScore": round(removal_penalty, 2),
        "recencyScore": round(recency_score, 2),
        "modelScore": round(model_score, 2),
        "activityScore": round(activity_score, 2),
        "rawRows": raw_rows,
        "validRows": valid_rows,
        "removedRows": removed_rows,
        "validDays": valid_days,
        "ageHours": None if age_hours is None else round(age_hours, 2),
    }
    return conf, details


def analyze_trend(daily: pd.DataFrame) -> Dict[str, Any]:
    result = {"trend": "stable", "icon": "stable", "feedDelta": 0.0, "waterDelta": 0.0}
    valid = daily[daily["valid_any_day"]].copy() if not daily.empty else pd.DataFrame()
    if len(valid) < 4:
        return result

    recent = valid.tail(3)
    previous = valid.iloc[-6:-3] if len(valid) >= 6 else valid.iloc[:-3]
    if previous.empty:
        return result

    def pct(cur, prev):
        cur = float(cur)
        prev = float(prev)
        if abs(prev) < 1e-6:
            return 0.0 if abs(cur) < 1e-6 else 100.0
        return _clamp(((cur - prev) / abs(prev)) * 100.0, -100.0, 100.0)

    fd = pct(recent["feed_target_kg"].mean(), previous["feed_target_kg"].mean())
    wd = pct(recent["water_target_l"].mean(), previous["water_target_l"].mean())
    result["feedDelta"] = round(fd, 2)
    result["waterDelta"] = round(wd, 2)

    if abs(fd) < 5 and abs(wd) < 5:
        result.update(trend="stable", icon="stable")
    elif fd > 5 or wd > 5:
        result.update(trend="increasing", icon="increasing")
    else:
        result.update(trend="decreasing", icon="decreasing")
    return result


def detect_anomaly(daily: pd.DataFrame) -> Dict[str, Any]:
    result = {"anomaly": False, "message": "", "value": 0.0}
    valid = daily[daily["valid_any_day"]].copy() if not daily.empty else pd.DataFrame()
    if len(valid) < 8:
        return result

    for col, label, unit in [
        ("feed_target_kg", "Feed consumption", "kg"),
        ("water_target_l", "Water consumption", "L"),
    ]:
        vals = valid[col].tail(30).astype(float)
        std = vals.std()
        if std <= 1e-9:
            continue
        z = abs((vals.iloc[-1] - vals.mean()) / std)
        if z > ANOMALY_Z:
            result.update({
                "anomaly": True,
                "message": f"{label} is unusual compared with recent data ({vals.iloc[-1]:.2f} {unit}).",
                "value": round(float(vals.iloc[-1]), 2),
            })
            return result
    return result


def build_schedule_predictions(target_date: pd.Timestamp, daily_feed: float, daily_water: float) -> List[Dict[str, Any]]:
    target_day = pd.to_datetime(target_date).normalize()
    n = len(FEED_SCHEDULE_HOURS)
    feed_each = max(0.0, float(daily_feed or 0.0)) / max(1, n)
    water_each = max(0.0, float(daily_water or 0.0)) / max(1, n)

    rows = []
    for hour in FEED_SCHEDULE_HOURS:
        slot = target_day + timedelta(hours=hour)
        rows.append({
            "date": str(slot.date()),
            "time": _schedule_label(slot),
            "hour": int(slot.hour),
            "feed_kg": round(feed_each, 2),
            "water_liters": round(water_each, 2),
        })
    return rows


def forecast_7_days(daily: pd.DataFrame, feed_info: Dict[str, Any], water_info: Dict[str, Any],
                    first_feed: float, first_water: float, start_date: pd.Timestamp) -> List[Dict[str, Any]]:
    rows = []
    sim = daily.copy()
    current_date = pd.to_datetime(start_date)

    for i in range(7):
        day = current_date + timedelta(days=i)
        feats = build_next_daily_features(sim, day)

        if i == 0:
            fv, wv = _round2(first_feed), _round2(first_water)
        else:
            fv, _ = predict_target(feed_info, sim, "feed_target_kg", feats, day.dayofweek)
            wv, _ = predict_target(water_info, sim, "water_target_l", feats, day.dayofweek)

        rows.append({
            "date": str(day.date()),
            "feed_kg": _round2(fv),
            "water_liters": _round2(wv),
        })

        # Append predicted day so later days can use lag/rolling features.
        new = {col: 0.0 for col in sim.columns}
        new.update({
            "date": day,
            "feed_target_kg": _round2(fv),
            "water_target_l": _round2(wv),
            "records": float(max(1.0, sim["records"].tail(3).median())) if not sim.empty else 1.0,
            "avg_flow": float(sim["avg_flow"].tail(3).median()) if not sim.empty else 0.0,
            "flow_active_rate": float(sim["flow_active_rate"].tail(3).median()) if not sim.empty else 0.0,
            "level_ok_rate": float(sim["level_ok_rate"].tail(3).median()) if not sim.empty else 1.0,
            "mean_feed_remaining": float(sim["mean_feed_remaining"].tail(3).median()) if not sim.empty else 0.0,
            "latest_water_total": float(sim["latest_water_total"].tail(1).iloc[-1]) if not sim.empty else 0.0,
            "day_index": float(len(sim)),
            "day_of_week": int(day.dayofweek),
            "month": int(day.month),
            "feed_prev_day": float(sim["feed_target_kg"].tail(1).iloc[-1]) if not sim.empty else 0.0,
            "water_prev_day": float(sim["water_target_l"].tail(1).iloc[-1]) if not sim.empty else 0.0,
            "roll3_feed_target": float(sim["feed_target_kg"].tail(3).median()) if not sim.empty else _round2(fv),
            "roll3_water_target": float(sim["water_target_l"].tail(3).median()) if not sim.empty else _round2(wv),
            "valid_feed_day": fv > 0,
            "valid_water_day": wv > 0,
            "valid_any_day": (fv > 0 or wv > 0),
        })
        sim = pd.concat([sim, pd.DataFrame([new])], ignore_index=True)

    return rows


def render_chart_b64(daily: pd.DataFrame, forecast_rows: List[Dict[str, Any]]) -> str:
    try:
        if daily.empty:
            return ""

        plot_df = daily.tail(60).copy()
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
        ax.set_facecolor(BG)

        ax.plot(plot_df["date"], plot_df["feed_target_kg"], color=CFEED, lw=2.0,
                marker="o", ms=3, label="Daily Feed Consumed")
        ax.plot(plot_df["date"], plot_df["water_target_l"], color=CWATER, lw=2.0,
                marker="o", ms=3, label="Daily Water Consumed")

        if forecast_rows:
            fdf = pd.DataFrame(forecast_rows)
            fdf["date"] = pd.to_datetime(fdf["date"])
            ax.axvline(x=plot_df["date"].iloc[-1], color=BORDER, lw=1, ls="--", alpha=0.6)
            ax.plot(fdf["date"], fdf["feed_kg"], color=CFEED, lw=1.8, ls="--", alpha=0.9,
                    label="Feed Forecast")
            ax.plot(fdf["date"], fdf["water_liters"], color=CWATER, lw=1.8, ls="--", alpha=0.9,
                    label="Water Forecast")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=25, color=MUTED, fontsize=8, ha="right")
        plt.yticks(color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color=BORDER, lw=0.6, ls="--", alpha=0.7)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                  frameon=False, labelcolor=MUTED, fontsize=8)
        ax.set_title("Daily Consumption and Forecast", color=MUTED, fontsize=11, pad=8)
        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor=BG)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[CHART] error: {e}")
        return ""


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def train_once():
    with _lock:
        if state["training"]:
            return False
        state.update(training=True, status="training", error="")

    db.write_ml_status("training", state.get("trained_rows", 0))

    try:
        readings = db.get_readings(limit=ROLLING_WINDOW)
        if not readings:
            db.write_ml_status("waiting", 0, "No data in Firebase yet")
            with _lock:
                state.update(training=False, status="waiting", error="No data in Firebase yet")
            return False

        raw_df, raw_stats = readings_to_professional_df(readings)
        raw_rows = int(raw_stats.get("rawRows", 0) or 0)
        valid_rows = int(raw_stats.get("validRows", 0) or 0)
        removed_rows = int(raw_stats.get("removedRows", 0) or 0)

        if raw_rows < MIN_ROWS:
            msg = f"Need {MIN_ROWS} rows, have {raw_rows}"
            db.write_ml_status("collecting", raw_rows, msg)
            with _lock:
                state.update(training=False, status="collecting", trained_rows=raw_rows,
                             valid_rows=valid_rows, removed_rows=removed_rows, error=msg)
            return False

        if raw_df.empty or valid_rows < MIN_ROWS:
            msg = f"Need {MIN_ROWS} valid sensor rows, have {valid_rows}"
            db.write_ml_status("collecting_valid_data", raw_rows, msg)
            with _lock:
                state.update(training=False, status="collecting_valid_data", trained_rows=raw_rows,
                             valid_rows=valid_rows, removed_rows=removed_rows, error=msg)
            return False

        event_df = add_consumption_columns(raw_df)
        daily = build_daily_consumption_table(event_df)
        valid_days = int(daily["valid_any_day"].sum()) if not daily.empty else 0

        print(f"[ML] Training rows={raw_rows}, valid_rows={valid_rows}, valid_days={valid_days}, removed={removed_rows}")

        if valid_days < MIN_VALID_DAYS_FOR_BASELINE:
            msg = "Collecting valid consumption data. Sensor rows exist, but no feed/water consumption change has been detected yet."
            db.write_ml_status("collecting_valid_data", raw_rows, msg)
            with _lock:
                state.update(training=False, status="collecting_valid_data", trained_rows=raw_rows,
                             valid_rows=valid_rows, valid_days=valid_days, removed_rows=removed_rows,
                             model_mode="waiting_for_consumption", error=msg)
            return False

        # Train feed and water target models using daily consumption targets.
        feed_info = train_target_model(daily, "feed_target_kg", "valid_feed_day")
        water_info = train_target_model(daily, "water_target_l", "valid_water_day")

        last_date = pd.to_datetime(daily["date"].max())
        pred_date = (last_date + timedelta(days=1)).normalize()
        next_features = build_next_daily_features(daily, pred_date)

        feed_v, feed_source = predict_target(feed_info, daily, "feed_target_kg", next_features, pred_date.dayofweek)
        water_v, water_source = predict_target(water_info, daily, "water_target_l", next_features, pred_date.dayofweek)

        # Final safety: if target is still zero but recent consumption exists, use recent consumption.
        if feed_v <= 0:
            feed_v = _round2(_robust_recent_average(daily["feed_target_kg"], 0.0, 3))
        if water_v <= 0:
            water_v = _round2(_robust_recent_average(daily["water_target_l"], 0.0, 3))

        # If both are still zero, do not call it ready.
        if feed_v <= 0 and water_v <= 0:
            msg = "No non-zero consumption target found after cleaning. Check load cell and water total readings."
            db.write_ml_status("collecting_valid_data", raw_rows, msg)
            with _lock:
                state.update(training=False, status="collecting_valid_data", trained_rows=raw_rows,
                             valid_rows=valid_rows, valid_days=valid_days, removed_rows=removed_rows,
                             model_mode="no_nonzero_target", error=msg)
            return False

        arima_feed = arima_forecast(daily["feed_target_kg"], feed_v)
        arima_water = arima_forecast(daily["water_target_l"], water_v)
        rows_7d = forecast_7_days(daily, feed_info, water_info, feed_v, water_v, pred_date)
        schedule_rows = build_schedule_predictions(pred_date, feed_v, water_v)
        next_sched = schedule_rows[0] if schedule_rows else {}

        conf, conf_details = calc_confidence(raw_stats, daily, feed_info, water_info)
        trend = analyze_trend(daily)
        anom = detect_anomaly(daily)

        model_mode = "professional_consumption_ml"
        if feed_info.get("mode") == "baseline" or water_info.get("mode") == "baseline":
            model_mode = "professional_consumption_baseline_blend"

        chart_b64 = render_chart_b64(daily, rows_7d)

        try:
            pat_day = daily.groupby("day_of_week")[["feed_target_kg", "water_target_l"]].mean().to_dict()
            pat_month = daily.groupby("month")[["feed_target_kg", "water_target_l"]].mean().to_dict()
            pat_sys = {"consumption": {
                "feed_target_kg": float(daily["feed_target_kg"].mean()),
                "water_target_l": float(daily["water_target_l"].mean()),
            }}
        except Exception:
            pat_day = pat_month = pat_sys = {}

        metrics = {
            "feedMAE": round(float(feed_info.get("metrics", {}).get("mae", 0.0)), 3),
            "feedRMSE": round(float(feed_info.get("metrics", {}).get("rmse", 0.0)), 3),
            "feedMAPE": round(float(feed_info.get("metrics", {}).get("mape", 0.0)), 2),
            "waterMAE": round(float(water_info.get("metrics", {}).get("mae", 0.0)), 3),
            "waterRMSE": round(float(water_info.get("metrics", {}).get("rmse", 0.0)), 3),
            "waterMAPE": round(float(water_info.get("metrics", {}).get("mape", 0.0)), 2),
        }

        ml_result = {
            # App-compatible keys
            "feedKg": feed_v,
            "waterL": water_v,
            "predDate": str(pred_date.date()),
            "arimaFeed": arima_feed,
            "arimaWater": arima_water,
            "confidence": conf,
            "confLabel": confidence_label(conf),
            "trend": trend["trend"],
            "trendIcon": trend["icon"],
            "feedDelta": trend["feedDelta"],
            "waterDelta": trend["waterDelta"],
            "anomaly": anom["anomaly"],
            "anomalyMsg": anom["message"],
            "modelRows": raw_rows,
            "trainedAt": datetime.utcnow().isoformat(),
            "patSystem": pat_sys,
            "patDay": pat_day,
            "patMonth": pat_month,
            "chartB64": chart_b64,
            "feedSchedule": schedule_rows,
            "nextFeedTime": next_sched.get("time", ""),
            "nextFeedDate": next_sched.get("date", ""),
            "nextFeedKg": next_sched.get("feed_kg", 0.0),
            "nextFeedWaterL": next_sched.get("water_liters", 0.0),

            # New professional ML details
            "predictionReady": True,
            "mlVersion": "professional_consumption_v2",
            "modelMode": model_mode,
            "targetType": "daily_consumption_target",
            "targetSource": f"feed:{feed_source}; water:{water_source}",
            "dataQuality": raw_stats,
            "confidenceDetails": conf_details,
            "validTrainingRows": valid_rows,
            "validTrainingDays": valid_days,
            "removedInvalidRows": removed_rows,
            "modelMetrics": metrics,
            "estimatedError": {
                "feedKgMAE": metrics["feedMAE"],
                "waterLMAE": metrics["waterMAE"],
                "meaning": "Average validation error based on recent holdout when enough valid days are available.",
            },
            "explanation": (
                "Prediction is based on cleaned feed/water consumption changes. "
                "Raw zero flow is allowed because chickens do not drink continuously."
            ),
        }

        ok1 = db.write_ml_result(ml_result)
        ok2 = db.write_forecast_7d(rows_7d)
        db.write_ml_status("ready", raw_rows)

        if anom["anomaly"]:
            db.push_alert("anomaly", anom["message"], anom["value"])

        with _lock:
            state.update(
                trained_rows=raw_rows,
                valid_rows=valid_rows,
                valid_days=valid_days,
                removed_rows=removed_rows,
                training=False,
                status="ready",
                error="",
                model_mode=model_mode,
                last_prediction={"feedKg": feed_v, "waterL": water_v, "confidence": conf},
            )

        print(
            f"[ML] ✅ feed={feed_v}kg water={water_v}L conf={int(conf*100)}% "
            f"mode={model_mode} valid_days={valid_days} write={'ok' if ok1 and ok2 else 'FAILED'}"
        )
        return True

    except Exception as ex:
        err = str(ex)[:250]
        print(f"[ML] ❌ {err}")
        print(traceback.format_exc())
        db.write_ml_status("error", state.get("trained_rows", 0), err)
        with _lock:
            state.update(training=False, status="error", error=err)
        return False


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def startup_train_once():
    try:
        time.sleep(STARTUP_TRAIN_DELAY)
        train_once()
    except Exception:
        print(traceback.format_exc())


def training_loop():
    last_trained_rows = 0
    last_train_attempt = 0
    print("[ML] Training loop started")

    while True:
        triggered = _event.wait(timeout=TRAIN_INTERVAL)
        _event.clear()
        try:
            with _lock:
                if state["training"]:
                    continue

            current = db.get_reading_count()
            if current < MIN_ROWS:
                msg = f"Need {MIN_ROWS} rows, have {current}"
                db.write_ml_status("collecting", current, msg)
                with _lock:
                    state.update(status="collecting", trained_rows=current, error=msg)
                continue

            now = time.time()
            new_rows = current - last_trained_rows
            due_by_rows = new_rows >= RETRAIN_EVERY or last_trained_rows == 0
            due_by_time = (now - last_train_attempt) >= TRAIN_INTERVAL
            if not (triggered or due_by_rows or due_by_time):
                continue

            last_train_attempt = now
            if train_once():
                last_trained_rows = db.get_reading_count()

        except Exception:
            print(traceback.format_exc())


def start_background_workers():
    global _workers_started
    with _lock:
        if _workers_started:
            return False
        _workers_started = True
    threading.Thread(target=training_loop, daemon=True).start()
    threading.Thread(target=startup_train_once, daemon=True).start()
    return True


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Poultry Farm Professional ML Server")


@app.on_event("startup")
async def on_startup():
    start_background_workers()


@app.get("/")
async def root():
    with _lock:
        s = dict(state)
    return JSONResponse({"service": "Poultry Farm Professional ML", "state": s})


@app.get("/health")
@app.head("/health")
async def health():
    return JSONResponse({"status": "ok", "ts": datetime.utcnow().isoformat()})


@app.get("/status")
async def get_status():
    with _lock:
        return JSONResponse(dict(state))


@app.post("/retrain")
@app.get("/retrain")
async def force_retrain():
    _event.set()
    return JSONResponse({"message": "Retrain queued", "note": "Check /status after a few seconds."})


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Poultry Farm Professional Cloud ML Server")
    print(f"  Firebase : {db.FIREBASE_URL}")
    print(f"  Retrains : every {RETRAIN_EVERY} rows or {TRAIN_INTERVAL}s")
    print(f"  Min rows : {MIN_ROWS}")
    print("  Target   : daily feed/water consumption")
    print("=" * 60)

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
