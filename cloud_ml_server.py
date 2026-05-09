from __future__ import annotations

import base64
import io
import math
import os
import threading
import time
import traceback
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

warnings.filterwarnings("ignore")

# Firebase helper layer from your project.
# This file still uses firebase_db.py for writing so it stays compatible with your APK.
import firebase_db as db

_env_url = os.getenv("FIREBASE_URL", "").strip()
if _env_url:
    db.FIREBASE_URL = _env_url
    print(f"[CONFIG] Firebase URL from env: {db.FIREBASE_URL}")
else:
    print(f"[CONFIG] Firebase URL from firebase_db.py: {db.FIREBASE_URL}")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
MIN_VALID_ROWS = int(os.getenv("MIN_VALID_ROWS", "50"))
RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY", "20"))
TRAIN_INTERVAL = int(os.getenv("TRAIN_INTERVAL", "120"))
STARTUP_TRAIN_DELAY = int(os.getenv("STARTUP_TRAIN_DELAY", "5"))

# 0 means no artificial limit. For prototype data (8,000+ rows), all data is fine.
# If your Firebase becomes very large later, set MAX_RAW_ROWS to 50000 or latest 60 days.
MAX_RAW_ROWS = int(os.getenv("MAX_RAW_ROWS", "0"))
FIREBASE_TIMEOUT = int(os.getenv("FIREBASE_TIMEOUT", "25"))

FEED_SCHEDULE_HOURS = [3, 8, 11, 14]
CHICKEN_COUNT_DEFAULT = int(os.getenv("CHICKEN_COUNT", "6"))

# Conservative sensor sanity limits for a small 6-chicken prototype.
# These are guard rails, not exact biological claims.
MAX_FEED_WEIGHT_KG = float(os.getenv("MAX_FEED_WEIGHT_KG", "20"))
MAX_INTERVAL_FEED_KG = float(os.getenv("MAX_INTERVAL_FEED_KG", "2.0"))
MAX_INTERVAL_WATER_L = float(os.getenv("MAX_INTERVAL_WATER_L", "5.0"))

# App-compatible output keys are preserved.
BG = "#0e1117"
BORDER = "#3d4257"
MUTED = "#a0aec0"
CFEED = "#50C8FF"
CWATER = "#1f77b4"

FEATURES = [
    "hour", "hour_sin", "hour_cos", "day_of_week", "month", "day_index",
    "chicken_count", "schedule_score", "is_schedule_hour", "flow_active_ratio",
    "level_low_ratio", "readings_count", "lag1_feed", "lag1_water",
    "roll3_feed", "roll3_water", "roll24_feed", "roll24_water",
]

SESSION = requests.Session()

state: Dict[str, Any] = {
    "status": "starting",
    "training": False,
    "raw_rows": 0,
    "valid_rows": 0,
    "valid_days": 0,
    "removed_rows": 0,
    "last_raw_count": 0,
    "trained_at": "",
    "error": "",
}
_lock = threading.Lock()
_event = threading.Event()

# ═════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _base_url() -> str:
    return db.FIREBASE_URL.rstrip("/")


def _safe_float(value: Any, default: float = 0.0, nonnegative: bool = True) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str):
            value = value.strip().replace("kg", "").replace("L/min", "").replace("L", "")
            if value.endswith("%"):
                value = value[:-1]
        n = float(value)
        if not np.isfinite(n):
            return default
        if nonnegative and n < 0:
            return 0.0
        return n
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default), nonnegative=False)))
    except Exception:
        return default


def _parse_date(rec: Dict[str, Any]) -> pd.Timestamp:
    ts_val = rec.get("timestamp")
    try:
        ts_num = float(ts_val)
        if ts_num > 1_000_000:
            dt = pd.to_datetime(ts_num, unit="s", errors="coerce")
            if not pd.isna(dt):
                return dt.tz_localize(None) if getattr(dt, "tz", None) is not None else dt
    except Exception:
        pass

    for key in ("ts", "date", "datetime", "createdAt"):
        raw = rec.get(key)
        if raw:
            dt = pd.to_datetime(str(raw).replace("Z", "+00:00"), errors="coerce")
            if not pd.isna(dt):
                try:
                    if dt.tzinfo is not None:
                        dt = dt.tz_convert(None)
                except Exception:
                    try:
                        dt = dt.tz_localize(None)
                    except Exception:
                        pass
                return dt

    return pd.NaT


def _is_error_status(value: Any) -> bool:
    txt = str(value or "").strip().lower()
    if not txt:
        return False
    bad_words = ["error", "missing", "not found", "failed", "disconnected", "invalid", "nan"]
    return any(w in txt for w in bad_words)


def _normalize_level(value: Any) -> str:
    txt = str(value or "").strip().lower()
    if txt in {"1", "true", "available", "ok", "normal", "full", "high", "water available"}:
        return "Available"
    if txt in {"0", "false", "low", "empty", "no water", "refill", "fill", "fill container"}:
        return "Low"
    if "low" in txt or "empty" in txt or "refill" in txt:
        return "Low"
    return str(value or "Available")


def _firebase_get_json(path: str, params: str = "") -> Optional[Any]:
    try:
        url = f"{_base_url()}/{path}.json{params}"
        r = SESSION.get(url, timeout=FIREBASE_TIMEOUT)
        if r.ok:
            return r.json()
        print(f"[DB GET] {path} failed: HTTP {r.status_code} {r.text[:120]}")
    except Exception as exc:
        print(f"[DB GET] {path} error: {exc}")
    return None


def _flatten_readings(raw: Any) -> List[Dict[str, Any]]:
    """Flatten Firebase payloads, including accidental /readings/readings nesting."""
    rows: List[Dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            # A real reading usually has timestamp/date and at least weight or totalLiters.
            has_time = any(k in obj for k in ("timestamp", "ts", "date"))
            has_sensor = any(k in obj for k in ("weight", "totalLiters", "flow", "level"))
            if has_time and has_sensor:
                rows.append(obj)
                return
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(raw)
    return rows


def fetch_all_readings() -> Tuple[List[Dict[str, Any]], int]:
    """Read all available /readings records; optional MAX_RAW_ROWS keeps latest N only."""
    if MAX_RAW_ROWS > 0:
        raw = _firebase_get_json("readings", f'?orderBy="timestamp"&limitToLast={MAX_RAW_ROWS}')
    else:
        # All historical readings. This is intended for thesis/prototype datasets.
        raw = _firebase_get_json("readings")

    if raw is None:
        # Fallback through firebase_db cache/query if direct fetch fails.
        # limit=0 means ALL readings, so no dataset is wasted by a count cap.
        try:
            if hasattr(db, "get_all_readings") and MAX_RAW_ROWS <= 0:
                readings = db.get_all_readings()
            else:
                fallback_limit = MAX_RAW_ROWS if MAX_RAW_ROWS > 0 else 0
                readings = db.get_readings(limit=fallback_limit)
            return readings, len(readings)
        except Exception:
            return [], 0

    readings = _flatten_readings(raw)
    readings.sort(key=lambda r: _parse_date(r) if not pd.isna(_parse_date(r)) else pd.Timestamp.min)
    if MAX_RAW_ROWS > 0 and len(readings) > MAX_RAW_ROWS:
        readings = readings[-MAX_RAW_ROWS:]
    return readings, len(readings)

# ═════════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING: RAW SENSOR READINGS → CONSUMPTION DATA
# ═════════════════════════════════════════════════════════════════════════════

def raw_readings_to_frame(readings: Iterable[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats = {
        "raw_rows": 0,
        "invalid_time": 0,
        "sensor_error_rows": 0,
        "invalid_value_rows": 0,
    }

    for rec in readings:
        stats["raw_rows"] += 1
        if not isinstance(rec, dict):
            stats["invalid_value_rows"] += 1
            continue

        dt = _parse_date(rec)
        if pd.isna(dt):
            stats["invalid_time"] += 1
            continue

        weight = _safe_float(rec.get("weight"), 0.0, True)
        total_liters = _safe_float(rec.get("totalLiters"), 0.0, True)
        flow = _safe_float(rec.get("flow"), 0.0, True)

        # Keep flow=0 as normal standby. Only exclude actual sensor-error statuses.
        if _is_error_status(rec.get("loadCellStatus")) or _is_error_status(rec.get("sensorStatus")):
            stats["sensor_error_rows"] += 1
            continue

        if weight > MAX_FEED_WEIGHT_KG or total_liters < 0 or flow < 0:
            stats["invalid_value_rows"] += 1
            continue

        chicken_count = _safe_int(rec.get("chickenCount", CHICKEN_COUNT_DEFAULT), CHICKEN_COUNT_DEFAULT)
        if chicken_count <= 0:
            chicken_count = CHICKEN_COUNT_DEFAULT

        rows.append({
            "date": pd.to_datetime(dt),
            "feed_weight": weight,
            "total_liters": total_liters,
            "flow": flow,
            "level": _normalize_level(rec.get("level", rec.get("waterStatus", "Available"))),
            "level_percent": _safe_float(rec.get("levelPercent"), np.nan, True),
            "chicken_count": chicken_count,
            "system": 1,
            "raw_mode": str(rec.get("mode", "sensor")),
        })

    if not rows:
        return pd.DataFrame(), stats

    df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    try:
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        pass
    return df, stats


def readings_to_consumption_df(readings: Iterable[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Converts raw records into hourly consumption records.

    Feed consumption = previous feed weight - current feed weight, only when positive.
    Water consumption = current totalLiters - previous totalLiters, only when positive.

    Important:
    - raw flow = 0.00 is NOT removed because chickens do not drink continuously.
    - sensor-error zero values are removed at raw filtering stage if status fields say error/missing.
    - valid zero-consumption hours are kept because inactivity is a real pattern.
    """
    raw_df, stats = raw_readings_to_frame(readings)
    if raw_df.empty:
        return raw_df, pd.DataFrame(), stats

    df = raw_df.copy()
    df["prev_weight"] = df["feed_weight"].shift(1)
    df["prev_water"] = df["total_liters"].shift(1)
    df["prev_date"] = df["date"].shift(1)

    # Do not calculate feed delta across large gaps or date resets/refill moments.
    gap_minutes = (df["date"] - df["prev_date"]).dt.total_seconds().div(60).fillna(0)
    same_reasonable_gap = (gap_minutes > 0) & (gap_minutes <= 180)

    feed_delta = (df["prev_weight"] - df["feed_weight"]).fillna(0)
    water_delta = (df["total_liters"] - df["prev_water"]).fillna(0)

    # Positive feed delta = consumed. Negative usually means refill/reset, so set to 0.
    feed_delta = feed_delta.where(feed_delta > 0, 0.0)
    feed_delta = feed_delta.where(feed_delta <= MAX_INTERVAL_FEED_KG, 0.0)
    feed_delta = feed_delta.where(same_reasonable_gap, 0.0)

    # Positive water delta = consumed/moved. Negative usually means sensor reset, so set to 0.
    water_delta = water_delta.where(water_delta > 0, 0.0)
    water_delta = water_delta.where(water_delta <= MAX_INTERVAL_WATER_L, 0.0)
    # Water can continue across midnight if totalLiters is cumulative; only remove unreasonable gaps.
    water_delta = water_delta.where(same_reasonable_gap, 0.0)

    df["feed_consumed"] = feed_delta.astype(float)
    df["water_consumed"] = water_delta.astype(float)
    df["hour_start"] = df["date"].dt.floor("h")
    df["flow_active"] = (df["flow"] > 0.001).astype(float)
    df["level_low"] = df["level"].astype(str).str.lower().str.contains("low|empty|refill|fill", regex=True).astype(float)

    hourly = df.groupby("hour_start", as_index=False).agg(
        feed_kg=("feed_consumed", "sum"),
        water_liters=("water_consumed", "sum"),
        flow=("flow", "mean"),
        flow_active_ratio=("flow_active", "mean"),
        level_low_ratio=("level_low", "mean"),
        readings_count=("date", "count"),
        chicken_count=("chicken_count", "median"),
        system=("system", "max"),
    )

    hourly = hourly.rename(columns={"hour_start": "date"}).sort_values("date").reset_index(drop=True)
    hourly["feed_kg"] = hourly["feed_kg"].clip(lower=0)
    hourly["water_liters"] = hourly["water_liters"].clip(lower=0)

    # Remove extreme hourly outliers likely caused by reset/wiring spikes.
    if len(hourly) >= 24:
        for col in ("feed_kg", "water_liters"):
            q99 = float(hourly[col].quantile(0.99))
            cap = max(q99 * 1.5, 0.05)
            hourly[col] = hourly[col].clip(upper=cap)

    # Keep valid zero-consumption hours, but remove fully empty leading rows if all values are zero and no readings.
    hourly = hourly[hourly["readings_count"] > 0].reset_index(drop=True)

    return raw_df, hourly, stats

# ═════════════════════════════════════════════════════════════════════════════
# FEATURES, METRICS, PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

def schedule_score_for_hour(hour: int) -> float:
    score = 0.0
    for h in FEED_SCHEDULE_HOURS:
        diff = min(abs(hour - h), 24 - abs(hour - h))
        score = max(score, math.exp(-((diff) ** 2) / (2 * (1.4 ** 2))))
    return float(score)


def is_schedule_hour(hour: int) -> int:
    return int(any(abs(hour - h) <= 1 for h in FEED_SCHEDULE_HOURS))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).reset_index(drop=True)

    out["hour"] = out["date"].dt.hour.astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["day_of_week"] = out["date"].dt.dayofweek.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    first_day = out["date"].dt.normalize().min()
    out["day_index"] = (out["date"].dt.normalize() - first_day).dt.days.astype(int)

    out["schedule_score"] = out["hour"].apply(schedule_score_for_hour)
    out["is_schedule_hour"] = out["hour"].apply(is_schedule_hour)

    for col in ("feed_kg", "water_liters", "flow_active_ratio", "level_low_ratio", "readings_count", "chicken_count"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["chicken_count"] = out["chicken_count"].replace(0, CHICKEN_COUNT_DEFAULT)

    out["lag1_feed"] = out["feed_kg"].shift(1).fillna(out["feed_kg"].mean())
    out["lag1_water"] = out["water_liters"].shift(1).fillna(out["water_liters"].mean())
    out["roll3_feed"] = out["feed_kg"].rolling(3, min_periods=1).mean()
    out["roll3_water"] = out["water_liters"].rolling(3, min_periods=1).mean()
    out["roll24_feed"] = out["feed_kg"].rolling(24, min_periods=1).mean()
    out["roll24_water"] = out["water_liters"].rolling(24, min_periods=1).mean()

    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100) if np.isfinite(denom).any() else 0.0
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(max(0.0, min(999.0, mape)), 2),
        "r2": round(max(-1.0, min(1.0, r2)), 4),
    }


def daily_consumption(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "feed_kg", "water_liters"])
    daily = df.copy()
    daily["day"] = pd.to_datetime(daily["date"]).dt.date
    out = daily.groupby("day", as_index=False).agg(
        feed_kg=("feed_kg", "sum"),
        water_liters=("water_liters", "sum"),
        rows=("feed_kg", "count"),
    )
    out["date"] = pd.to_datetime(out["day"].astype(str))
    return out[["date", "feed_kg", "water_liters", "rows"]]


def baseline_daily_targets(df: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    ddf = daily_consumption(df)
    if ddf.empty:
        return 0.0, 0.0, ddf

    # Prefer recent complete/near-complete days. With hourly data, a full day has close to 24 rows.
    valid_days = ddf[ddf["rows"] >= max(6, int(ddf["rows"].median() * 0.5))].copy()
    if valid_days.empty:
        valid_days = ddf.copy()
    recent = valid_days.tail(7) if len(valid_days) >= 7 else valid_days

    feed = float(recent["feed_kg"].replace([np.inf, -np.inf], np.nan).dropna().mean())
    water = float(recent["water_liters"].replace([np.inf, -np.inf], np.nan).dropna().mean())
    return round(max(0.0, feed), 3), round(max(0.0, water), 3), ddf


def build_future_feature_row(history: pd.DataFrame, slot: pd.Timestamp) -> Dict[str, Any]:
    hist = history.copy()
    last = hist.iloc[-1]
    hour = int(slot.hour)
    first_day = pd.to_datetime(hist["date"]).dt.normalize().min()
    return {
        "date": slot,
        "hour": hour,
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "day_of_week": int(slot.dayofweek),
        "month": int(slot.month),
        "day_index": int((slot.normalize() - first_day).days),
        "chicken_count": float(last.get("chicken_count", CHICKEN_COUNT_DEFAULT) or CHICKEN_COUNT_DEFAULT),
        "schedule_score": schedule_score_for_hour(hour),
        "is_schedule_hour": is_schedule_hour(hour),
        "flow_active_ratio": float(hist["flow_active_ratio"].tail(24).mean()) if "flow_active_ratio" in hist else 0.0,
        "level_low_ratio": float(hist["level_low_ratio"].tail(24).mean()) if "level_low_ratio" in hist else 0.0,
        "readings_count": float(hist["readings_count"].tail(24).median()) if "readings_count" in hist else 1.0,
        "lag1_feed": float(last.get("feed_kg", 0.0)),
        "lag1_water": float(last.get("water_liters", 0.0)),
        "roll3_feed": float(hist["feed_kg"].tail(3).mean()),
        "roll3_water": float(hist["water_liters"].tail(3).mean()),
        "roll24_feed": float(hist["feed_kg"].tail(24).mean()),
        "roll24_water": float(hist["water_liters"].tail(24).mean()),
    }


def train_models(df_feat: pd.DataFrame) -> Tuple[Any, Any, Dict[str, Any]]:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = df_feat[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_feed = df_feat["feed_kg"].astype(float)
    y_water = df_feat["water_liters"].astype(float)

    def make_model() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingRegressor(
                n_estimators=160,
                max_depth=3,
                learning_rate=0.045,
                subsample=0.90,
                random_state=42,
            )),
        ])

    metrics: Dict[str, Any] = {}
    n = len(df_feat)
    split = int(n * 0.80)
    if n >= 80 and split < n:
        m_feed_test = make_model()
        m_water_test = make_model()
        m_feed_test.fit(X.iloc[:split], y_feed.iloc[:split])
        m_water_test.fit(X.iloc[:split], y_water.iloc[:split])
        pred_feed = np.maximum(0.0, m_feed_test.predict(X.iloc[split:]))
        pred_water = np.maximum(0.0, m_water_test.predict(X.iloc[split:]))
        metrics["feed"] = _metrics(y_feed.iloc[split:].values, pred_feed)
        metrics["water"] = _metrics(y_water.iloc[split:].values, pred_water)
    else:
        metrics["feed"] = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}
        metrics["water"] = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}

    m_feed = make_model()
    m_water = make_model()
    m_feed.fit(X, y_feed)
    m_water.fit(X, y_water)
    return m_feed, m_water, metrics


def predict_day_hourly(df_feat: pd.DataFrame, model_feed: Any, model_water: Any, target_day: pd.Timestamp) -> Tuple[pd.DataFrame, float, float]:
    tmp = df_feat.copy().sort_values("date").reset_index(drop=True)
    target_day = pd.to_datetime(target_day).normalize()
    rows: List[Dict[str, Any]] = []

    # caps are learned from history to prevent one noisy value from dominating daily target.
    avg_feed, avg_water, _daily = baseline_daily_targets(tmp)
    hourly_feed_cap = max(0.05, float(tmp["feed_kg"].quantile(0.98)) * 2.0)
    hourly_water_cap = max(0.05, float(tmp["water_liters"].quantile(0.98)) * 2.0)

    for hr in range(24):
        slot = target_day + timedelta(hours=hr)
        feat = build_future_feature_row(tmp, slot)
        X_next = pd.DataFrame([feat])[FEATURES].fillna(0.0)
        feed_pred = float(model_feed.predict(X_next)[0])
        water_pred = float(model_water.predict(X_next)[0])
        feed_pred = max(0.0, min(feed_pred, hourly_feed_cap))
        water_pred = max(0.0, min(water_pred, hourly_water_cap))

        row = dict(feat)
        row["feed_kg"] = feed_pred
        row["water_liters"] = water_pred
        rows.append(row)
        tmp = pd.concat([tmp, pd.DataFrame([row])], ignore_index=True)

    hourly = pd.DataFrame(rows)
    model_feed_day = float(hourly["feed_kg"].sum())
    model_water_day = float(hourly["water_liters"].sum())

    # Professional guardrail: blend ML with recent average so valid data never produces false 0.
    # If the model output is too low/high relative to baseline, the baseline keeps result realistic.
    if avg_feed > 0:
        low, high = avg_feed * 0.45, avg_feed * 1.75
        model_feed_day = min(max(model_feed_day, low), high)
    if avg_water > 0:
        low, high = avg_water * 0.45, avg_water * 1.85
        model_water_day = min(max(model_water_day, low), high)

    feed_day = round((model_feed_day * 0.70) + (avg_feed * 0.30), 2) if avg_feed > 0 else round(model_feed_day, 2)
    water_day = round((model_water_day * 0.70) + (avg_water * 0.30), 2) if avg_water > 0 else round(model_water_day, 2)

    return hourly, max(0.0, feed_day), max(0.0, water_day)


def compute_schedule_rows(df_feat: pd.DataFrame, target_day: pd.Timestamp, daily_feed: float, daily_water: float) -> List[Dict[str, Any]]:
    # Learn schedule proportions from historical consumption around schedule hours.
    hist = df_feat.copy()
    hist["hour"] = pd.to_datetime(hist["date"]).dt.hour

    weights_feed: List[float] = []
    weights_water: List[float] = []
    for h in FEED_SCHEDULE_HOURS:
        mask = hist["hour"].between(max(0, h - 1), min(23, h + 2))
        weights_feed.append(float(hist.loc[mask, "feed_kg"].sum()))
        weights_water.append(float(hist.loc[mask, "water_liters"].sum()))

    def fractions(values: List[float]) -> List[float]:
        s = sum(max(0.0, v) for v in values)
        if s <= 1e-9:
            return [1.0 / len(values)] * len(values)
        # Avoid a schedule becoming exactly 0 due to sparse data.
        raw = [max(0.05, v / s) for v in values]
        total = sum(raw)
        return [v / total for v in raw]

    f_frac = fractions(weights_feed)
    w_frac = fractions(weights_water)

    target_day = pd.to_datetime(target_day).normalize()
    rows: List[Dict[str, Any]] = []
    for i, h in enumerate(FEED_SCHEDULE_HOURS):
        slot = target_day + timedelta(hours=h)
        rows.append({
            "date": str(slot.date()),
            "time": slot.strftime("%I:%M %p").lstrip("0"),
            "hour": int(h),
            "feed_kg": round(max(0.0, daily_feed * f_frac[i]), 2),
            "water_liters": round(max(0.0, daily_water * w_frac[i]), 2),
        })

    # Ensure rounded schedule values add close to daily totals; small rounding differences are acceptable.
    return rows


def compute_arima_daily(daily_df: pd.DataFrame, fallback_feed: float, fallback_water: float) -> Tuple[float, float]:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        if len(daily_df) < 5:
            return fallback_feed, fallback_water
        series = daily_df.set_index("date")[["feed_kg", "water_liters"]].asfreq("D").fillna(0)

        feed_val = fallback_feed
        water_val = fallback_water
        try:
            if series["feed_kg"].nunique() > 1:
                feed_val = float(ARIMA(series["feed_kg"].values, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit().forecast(1)[0])
        except Exception:
            pass
        try:
            if series["water_liters"].nunique() > 1:
                water_val = float(ARIMA(series["water_liters"].values, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False).fit().forecast(1)[0])
        except Exception:
            pass

        return round(max(0.0, feed_val), 2), round(max(0.0, water_val), 2)
    except Exception:
        return fallback_feed, fallback_water


def analyze_trend(daily_df: pd.DataFrame) -> Dict[str, Any]:
    result = {"trend": "stable", "icon": "stable", "feedDelta": 0.0, "waterDelta": 0.0}
    if daily_df.empty or len(daily_df) < 4:
        return result
    recent = daily_df.tail(3)
    prev = daily_df.iloc[-6:-3] if len(daily_df) >= 6 else daily_df.iloc[:-3]
    if prev.empty:
        return result

    def pct(cur: float, old: float) -> float:
        if abs(old) < 1e-9:
            return 0.0 if abs(cur) < 1e-9 else 100.0
        return max(-100.0, min(100.0, ((cur - old) / abs(old)) * 100.0))

    fd = pct(float(recent["feed_kg"].mean()), float(prev["feed_kg"].mean()))
    wd = pct(float(recent["water_liters"].mean()), float(prev["water_liters"].mean()))
    result["feedDelta"] = round(fd, 2)
    result["waterDelta"] = round(wd, 2)

    if abs(fd) < 5 and abs(wd) < 5:
        result.update(trend="stable", icon="stable")
    elif fd > 5 or wd > 5:
        result.update(trend="increasing", icon="increasing")
    else:
        result.update(trend="decreasing", icon="decreasing")
    return result


def detect_anomaly(daily_df: pd.DataFrame) -> Dict[str, Any]:
    if daily_df.empty or len(daily_df) < 7:
        return {"anomaly": False, "message": "", "value": 0.0}
    base = daily_df.iloc[:-1]
    last = daily_df.iloc[-1]
    for col, label in [("feed_kg", "Feed consumption"), ("water_liters", "Water consumption")]:
        std = float(base[col].std() or 0.0)
        mean = float(base[col].mean() or 0.0)
        if std <= 1e-9:
            continue
        z = abs((float(last[col]) - mean) / std)
        if z >= 2.8:
            return {
                "anomaly": True,
                "message": f"{label} is outside the usual pattern ({z:.1f} standard deviations).",
                "value": round(float(last[col]), 3),
            }
    return {"anomaly": False, "message": "", "value": 0.0}


def calc_confidence(valid_rows: int, valid_days: int, raw_rows: int, removed_rows: int, metrics: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
    row_score = min(1.0, valid_rows / 500.0)
    day_score = min(1.0, valid_days / 14.0)
    quality_score = 1.0 - min(0.6, removed_rows / max(1, raw_rows))

    feed_mape = float(metrics.get("feed", {}).get("mape", 0.0) or 0.0)
    water_mape = float(metrics.get("water", {}).get("mape", 0.0) or 0.0)
    avg_mape = (feed_mape + water_mape) / 2.0
    if avg_mape <= 0:
        model_score = 0.65 if valid_rows >= 80 else 0.45
    else:
        model_score = max(0.15, 1.0 - min(avg_mape, 100.0) / 100.0)

    conf = (row_score * 0.30) + (day_score * 0.25) + (quality_score * 0.20) + (model_score * 0.25)
    conf = round(max(0.05, min(0.98, conf)), 2)
    if conf >= 0.80:
        label = "High"
    elif conf >= 0.55:
        label = "Medium"
    elif conf >= 0.30:
        label = "Low"
    else:
        label = "Very Low"
    return conf, label, {
        "rowScore": round(row_score, 3),
        "dayScore": round(day_score, 3),
        "qualityScore": round(quality_score, 3),
        "modelScore": round(model_score, 3),
    }


def pattern_tables(df_feat: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    try:
        pat_sys = {"1": {
            "feed_kg": round(float(df_feat["feed_kg"].mean()), 4),
            "water_liters": round(float(df_feat["water_liters"].mean()), 4),
        }}
        pat_day = df_feat.groupby("day_of_week")[["feed_kg", "water_liters"]].mean().round(4).to_dict()
        pat_month = df_feat.groupby("month")[["feed_kg", "water_liters"]].mean().round(4).to_dict()
        return pat_sys, pat_day, pat_month
    except Exception:
        return {}, {}, {}


def render_chart_b64(daily_df: pd.DataFrame, forecast_rows: List[Dict[str, Any]]) -> str:
    try:
        if daily_df.empty:
            return ""
        plot_df = daily_df.tail(30).copy()
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
        ax.set_facecolor(BG)
        ax.plot(plot_df["date"], plot_df["feed_kg"], color=CFEED, lw=2, marker="o", ms=3, label="Feed Consumption")
        ax.plot(plot_df["date"], plot_df["water_liters"], color=CWATER, lw=2, marker="o", ms=3, label="Water Consumption")
        if forecast_rows:
            fdf = pd.DataFrame(forecast_rows)
            fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
            ax.plot(fdf["date"], fdf["feed_kg"], color=CFEED, lw=1.8, ls="--", alpha=0.85, label="Feed Forecast")
            ax.plot(fdf["date"], fdf["water_liters"], color=CWATER, lw=1.8, ls="--", alpha=0.85, label="Water Forecast")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=25, color=MUTED, fontsize=8, ha="right")
        plt.yticks(color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color=BORDER, lw=0.6, ls="--", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, labelcolor=MUTED, fontsize=8)
        ax.set_title("Daily Feed and Water Consumption Trend", color=MUTED, fontsize=11, pad=8)
        plt.tight_layout(pad=1.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor=BG)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        print(f"[CHART] error: {exc}")
        return ""

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING CYCLE
# ═════════════════════════════════════════════════════════════════════════════

def train_once() -> bool:
    with _lock:
        if state.get("training"):
            return False
        state.update(training=True, status="training", error="")

    try:
        db.write_ml_status("training", int(state.get("valid_rows", 0)))
    except Exception:
        pass

    try:
        readings, raw_count = fetch_all_readings()
        if not readings:
            msg = "No data in Firebase /readings yet."
            db.write_ml_status("waiting", 0, msg, rawRows=0, validRows=0, trainingMode="continuous_all_valid_historical_data")
            with _lock:
                state.update(training=False, status="waiting", raw_rows=0, valid_rows=0, error=msg)
            return False

        raw_df, hourly_df, prep_stats = readings_to_consumption_df(readings)
        raw_rows = int(prep_stats.get("raw_rows", raw_count))
        valid_rows = int(len(hourly_df))
        valid_days = int(hourly_df["date"].dt.date.nunique()) if not hourly_df.empty else 0
        removed_rows = max(0, raw_rows - int(len(raw_df)))

        if hourly_df.empty or valid_rows < MIN_VALID_ROWS:
            msg = f"Collecting valid consumption data: need {MIN_VALID_ROWS} hourly rows, have {valid_rows}."
            db.write_ml_status("collecting", valid_rows, msg, rawRows=raw_rows, validRows=valid_rows, validDays=valid_days, removedRows=removed_rows, trainingMode="continuous_all_valid_historical_data")
            with _lock:
                state.update(
                    training=False,
                    status="collecting",
                    raw_rows=raw_rows,
                    valid_rows=valid_rows,
                    valid_days=valid_days,
                    removed_rows=removed_rows,
                    error=msg,
                )
            return False

        print(f"[ML] Training on all valid data: raw={raw_rows}, hourly_valid={valid_rows}, days={valid_days}")

        df_feat = add_features(hourly_df)
        model_feed, model_water, metrics = train_models(df_feat)

        last_date = pd.to_datetime(df_feat["date"].max())
        target_day = (last_date + timedelta(days=1)).normalize()

        _hourly_next, feed_target, water_target = predict_day_hourly(df_feat, model_feed, model_water, target_day)
        base_feed, base_water, daily_df = baseline_daily_targets(df_feat)

        # If valid data exists but model somehow returns 0, use recent baseline instead of false zero.
        if feed_target <= 0 and base_feed > 0:
            feed_target = round(base_feed, 2)
        if water_target <= 0 and base_water > 0:
            water_target = round(base_water, 2)

        arima_feed, arima_water = compute_arima_daily(daily_df, feed_target, water_target)

        forecast_rows: List[Dict[str, Any]] = []
        for i in range(7):
            f_day = target_day + timedelta(days=i)
            _h, f_feed, f_water = predict_day_hourly(df_feat, model_feed, model_water, f_day)
            forecast_rows.append({
                "date": str(f_day.date()),
                "feed_kg": round(max(0.0, f_feed), 2),
                "water_liters": round(max(0.0, f_water), 2),
            })

        schedule_rows = compute_schedule_rows(df_feat, target_day, feed_target, water_target)
        next_sched = schedule_rows[0] if schedule_rows else {}

        trend = analyze_trend(daily_df)
        anomaly = detect_anomaly(daily_df)
        confidence, confidence_label, confidence_details = calc_confidence(valid_rows, valid_days, raw_rows, removed_rows, metrics)
        pat_sys, pat_day, pat_month = pattern_tables(df_feat)
        chart_b64 = render_chart_b64(daily_df, forecast_rows)

        feed_mae = float(metrics.get("feed", {}).get("mae", 0.0) or 0.0)
        water_mae = float(metrics.get("water", {}).get("mae", 0.0) or 0.0)

        ml_result = {
            # App-compatible main values
            "feedKg": round(float(feed_target), 2),
            "waterL": round(float(water_target), 2),
            "predDate": str(target_day.date()),
            "arimaFeed": round(float(arima_feed), 2),
            "arimaWater": round(float(arima_water), 2),
            "confidence": confidence,
            "confLabel": confidence_label,
            "trend": trend["trend"],
            "trendIcon": trend["icon"],
            "feedDelta": trend["feedDelta"],
            "waterDelta": trend["waterDelta"],
            "anomaly": anomaly["anomaly"],
            "anomalyMsg": anomaly["message"],
            "modelRows": valid_rows,
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

            # Professional thesis/debug details
            "targetType": "daily_feed_and_water_consumption_target",
            "trainingMode": "continuous_all_valid_historical_data",
            "dataMode": "consumption_based",
            "rawRows": raw_rows,
            "validRows": valid_rows,
            "validDays": valid_days,
            "removedRows": removed_rows,
            "minimumValidRows": MIN_VALID_ROWS,
            "chickenCount": CHICKEN_COUNT_DEFAULT,
            "baselineFeedKg": round(float(base_feed), 2),
            "baselineWaterL": round(float(base_water), 2),
            "metrics": metrics,
            "feedMAE": round(feed_mae, 4),
            "waterMAE": round(water_mae, 4),
            "estimatedErrorFeedKg": round(feed_mae * 24, 3),
            "estimatedErrorWaterL": round(water_mae * 24, 3),
            "confidenceDetails": confidence_details,
            "preprocessing": {
                "rawRows": raw_rows,
                "acceptedRawRows": int(len(raw_df)),
                "removedRows": removed_rows,
                "invalidTimeRows": int(prep_stats.get("invalid_time", 0)),
                "sensorErrorRows": int(prep_stats.get("sensor_error_rows", 0)),
                "invalidValueRows": int(prep_stats.get("invalid_value_rows", 0)),
                "flowZeroPolicy": "Flow 0.00 is kept as normal standby, not sensor error.",
            },
        }

        ok1 = db.write_ml_result(ml_result)
        ok2 = db.write_forecast_7d(forecast_rows)
        db.write_ml_status("ready", valid_rows, "", rawRows=raw_rows, validRows=valid_rows, validDays=valid_days, removedRows=removed_rows, trainingMode="continuous_all_valid_historical_data", dataMode="hourly_consumption")
        if anomaly["anomaly"]:
            db.push_alert("anomaly", anomaly["message"], anomaly["value"])

        with _lock:
            state.update(
                training=False,
                status="ready",
                raw_rows=raw_rows,
                valid_rows=valid_rows,
                valid_days=valid_days,
                removed_rows=removed_rows,
                last_raw_count=raw_rows,
                trained_at=ml_result["trainedAt"],
                error="" if ok1 and ok2 else "Firebase write failed",
            )

        print(
            f"[ML] READY feed={feed_target:.2f} kg water={water_target:.2f} L "
            f"conf={confidence_label} rows={valid_rows} days={valid_days} "
            f"firebase_write={'ok' if ok1 and ok2 else 'FAILED'}"
        )
        return bool(ok1 and ok2)

    except Exception as exc:
        err = str(exc)[:300]
        print(f"[ML] ERROR: {err}")
        print(traceback.format_exc())
        try:
            db.write_ml_status("error", int(state.get("valid_rows", 0)), err)
        except Exception:
            pass
        with _lock:
            state.update(training=False, status="error", error=err)
        return False

# ═════════════════════════════════════════════════════════════════════════════
# CONTINUOUS TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def startup_train_once() -> None:
    time.sleep(STARTUP_TRAIN_DELAY)
    train_once()


def training_loop() -> None:
    print("[ML] Continuous all-data training loop started")
    last_seen_count = 0
    while True:
        triggered = _event.wait(timeout=TRAIN_INTERVAL)
        _event.clear()
        try:
            with _lock:
                if state.get("training"):
                    continue

            readings, raw_count = fetch_all_readings()
            if raw_count <= 0:
                db.write_ml_status("waiting", 0, "No data in Firebase /readings yet.", rawRows=0, validRows=0, trainingMode="continuous_all_valid_historical_data")
                with _lock:
                    state.update(status="waiting", raw_rows=0)
                continue

            new_rows = raw_count - last_seen_count
            should_train = triggered or last_seen_count == 0 or new_rows >= RETRAIN_EVERY
            if should_train:
                train_once()
                with _lock:
                    last_seen_count = int(state.get("raw_rows", raw_count) or raw_count)
            else:
                with _lock:
                    state.update(status=state.get("status", "ready"), raw_rows=raw_count)
        except Exception:
            print(traceback.format_exc())

# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Poultry Farm ML Server", version="continuous-consumption-v3")


@app.get("/")
async def root():
    with _lock:
        s = dict(state)
    return JSONResponse({
        "service": "Poultry Farm ML Server",
        "version": "continuous-consumption-v3",
        "description": "Trains on all valid historical readings and predicts daily feed/water consumption targets.",
        "state": s,
    })


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "ts": datetime.utcnow().isoformat()})


@app.get("/status")
async def get_status():
    with _lock:
        return JSONResponse(dict(state))


@app.get("/debug-data")
async def debug_data():
    readings, raw_count = fetch_all_readings()
    raw_df, hourly_df, stats = readings_to_consumption_df(readings)
    days = int(hourly_df["date"].dt.date.nunique()) if not hourly_df.empty else 0
    first_date = str(raw_df["date"].min()) if not raw_df.empty else ""
    last_date = str(raw_df["date"].max()) if not raw_df.empty else ""
    return JSONResponse({
        "rawRowsFetched": raw_count,
        "firebaseReadingCount": db.get_reading_count() if hasattr(db, "get_reading_count") else raw_count,
        "acceptedRawRows": int(len(raw_df)),
        "validHourlyConsumptionRows": int(len(hourly_df)),
        "validDays": days,
        "firstDate": first_date,
        "lastDate": last_date,
        "stats": stats,
        "policy": "Uses all valid historical data; converts raw readings to feed/water consumption before ML training.",
    })


@app.get("/retrain")
async def retrain_get():
    # Browser-friendly retrain trigger.
    _event.set()
    return JSONResponse({"message": "Retrain triggered. Check /status and Firebase /ml_result after a few seconds."})


@app.post("/retrain")
async def retrain_post():
    _event.set()
    return JSONResponse({"message": "Retrain triggered. Check /status and Firebase /ml_result after a few seconds."})


@app.get("/train-now")
async def train_now():
    # Runs immediately in the request. Useful for testing only; may take longer for large data.
    ok = train_once()
    with _lock:
        s = dict(state)
    return JSONResponse({"ok": ok, "state": s})

# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Poultry Farm Cloud ML Server - Continuous Consumption Version")
    print(f"  Firebase      : {db.FIREBASE_URL}")
    print(f"  Training mode : ALL valid historical data")
    print(f"  Target type   : daily feed/water consumption")
    print(f"  Retrain       : every {RETRAIN_EVERY} new rows or {TRAIN_INTERVAL}s")
    print(f"  Min rows      : {MIN_VALID_ROWS} valid hourly consumption rows")
    print(f"  Schedule      : {FEED_SCHEDULE_HOURS}")
    print("=" * 70)

    threading.Thread(target=training_loop, daemon=True).start()
    threading.Thread(target=startup_train_once, daemon=True).start()

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
