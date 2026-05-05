import os
import sys
import time
import threading
import traceback
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd
import io
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

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
RETRAIN_EVERY  = 20     
MIN_ROWS       = 50     
MIN_DAILY_ROWS = 3      
ROLLING_WINDOW = 1000   
TRAIN_INTERVAL = 120    
STARTUP_TRAIN_DELAY = 5
FEED_SCHEDULE_HOURS = [3, 8, 11, 14]  
ANOMALY_Z      = 2.5    

BG      = "#0e1117"
BORDER  = "#3d4257"
MUTED   = "#a0aec0"
CFEED   = "#50C8FF"
CWATER  = "#1f77b4"


FEATURES = [
    "water_liters", "system", "day_of_week", "month",
    "hour", "lag1_feed", "lag1_water", "roll3_feed",
]

# ═════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STATE
# ═════════════════════════════════════════════════════════════════════════════
state: Dict[str, Any] = {
    "trained_rows": 0,
    "training":     False,
    "status":       "starting",
    "error":        "",
    "last_attempt_at": "",
    "last_trained_at": "",
    "last_write_ok": False,
    "training_source": "",
    "last_anomaly_key": "",
    "last_anomaly_at": 0.0,
}
_lock  = threading.Lock()
_event = threading.Event()
_workers_started = False


# ═════════════════════════════════════════════════════════════════════════════
# ML ANALYSIS HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def detect_anomaly(df: pd.DataFrame) -> Dict:
    result = {"anomaly": False, "message": "", "value": 0.0}
    if len(df) < 10:
        return result
    for col, label in [("feed_kg", "Feed/Weight"), ("water_liters", "Water")]:
        vals = df[col].tail(50).dropna()
        if len(vals) < 5:
            continue
        z = abs((vals.iloc[-1] - vals.mean()) / (vals.std() + 1e-9))
        if z > ANOMALY_Z:
            result.update(
                anomaly=True,
                message=f"⚠️ {label} reading is {z:.1f}σ from normal ({vals.iloc[-1]:.3f})",
                value=float(vals.iloc[-1]),
            )
            return result
    return result


def analyze_trend(df: pd.DataFrame) -> Dict:
    result = {"trend": "stable", "icon": "✅", "feedDelta": 0.0, "waterDelta": 0.0}
    if len(df) < 40:
        return result
    rec  = df.tail(20)
    prev = df.iloc[-40:-20]

    def safe_pct_change(current_mean, previous_mean):
        """Avoid huge percent values when previous average is 0.00 or near zero."""
        try:
            current_mean = float(current_mean)
            previous_mean = float(previous_mean)
            if abs(previous_mean) < 0.01:
                if abs(current_mean) < 0.01:
                    return 0.0
                return 100.0 if current_mean > previous_mean else -100.0
            value = ((current_mean - previous_mean) / abs(previous_mean)) * 100.0
            return max(-100.0, min(100.0, value))
        except Exception:
            return 0.0

    fd = safe_pct_change(rec["feed_kg"].mean(), prev["feed_kg"].mean())
    wd = safe_pct_change(rec["water_liters"].mean(), prev["water_liters"].mean())
    result["feedDelta"]  = round(fd, 2)
    result["waterDelta"] = round(wd, 2)
    if detect_anomaly(df)["anomaly"]:
        result.update(trend="warning", icon="🚨")
    elif abs(fd) < 5 and abs(wd) < 5:
        result.update(trend="stable",     icon="✅")
    elif fd > 5 or wd > 5:
        result.update(trend="increasing", icon="📈")
    else:
        result.update(trend="decreasing", icon="📉")
    return result


def calc_confidence(df: pd.DataFrame, rows: int) -> float:
    row_score = min(1.0, max(0.0, (rows - MIN_ROWS) / max(1, 500 - MIN_ROWS)))
    try:
        cv = (df["feed_kg"].std()      / (df["feed_kg"].mean()      + 1e-9) +
              df["water_liters"].std() / (df["water_liters"].mean() + 1e-9)) / 2
        var_score = max(0.0, 1.0 - cv)
    except Exception:
        var_score = 0.5
    return round(min(1.0, max(0.05, row_score * 0.6 + var_score * 0.4)), 2)


def confidence_label(c: float) -> str:
    if c >= 0.80: return "High"
    if c >= 0.55: return "Medium"
    if c >= 0.30: return "Low"
    return "Very Low"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to DataFrame.
    Called once during training and once per prediction row.
    Single source of truth — prevents train/predict feature mismatch.
    """
    df = df.copy()
    df["hour"]       = pd.to_datetime(df["date"]).dt.hour
    df["lag1_feed"]  = df["feed_kg"].shift(1).fillna(df["feed_kg"].mean())
    df["lag1_water"] = df["water_liters"].shift(1).fillna(df["water_liters"].mean())
    df["roll3_feed"] = df["feed_kg"].rolling(3, min_periods=1).mean()
    return df


def build_predict_row(last_row: pd.Series, next_date: pd.Timestamp,
                      recent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single prediction input row with all FEATURES.
    Uses last_row values for lag features.
    """
    return pd.DataFrame([{
        "water_liters": float(last_row["water_liters"]),
        "system":       int(last_row.get("system", 1)),
        "day_of_week":  next_date.weekday(),
        "month":        next_date.month,
        "hour":         0,
        "lag1_feed":    float(last_row["feed_kg"]),
        "lag1_water":   float(last_row["water_liters"]),
        "roll3_feed":   float(recent_df["feed_kg"].tail(3).mean()),
    }])

def _schedule_label(dt: pd.Timestamp) -> str:
    """Human-readable feeding schedule label."""
    try:
        return dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return str(dt)


def build_schedule_predictions(last_row: pd.Series, recent_df: pd.DataFrame,
                               m_feed, m_water, count: int = 4,
                               target_date: pd.Timestamp = None,
                               daily_feed: float = None,
                               daily_water: float = None) -> list:
    """
    Build the full one-day feeding schedule shown in the APK.

    AI Prediction = one-day target.
    Scheduled Prediction = the same one-day target divided into the fixed
    feeding times: 3:00 AM, 8:00 AM, 11:00 AM, and 2:00 PM.

    This avoids confusing partial schedules (only 1 or 2 rows) and keeps the
    schedule total equal to the daily AI target when daily_feed/daily_water are
    provided.
    """
    rows = []
    try:
        base_dt = pd.to_datetime(last_row["date"])
        target_day = pd.to_datetime(target_date).normalize() if target_date is not None else (base_dt + timedelta(days=1)).normalize()
        schedule_hours = FEED_SCHEDULE_HOURS[:count]
        candidates = [target_day + timedelta(hours=hr) for hr in schedule_hours]

        if daily_feed is not None or daily_water is not None:
            total_feed = max(0.0, float(daily_feed or 0.0))
            total_water = max(0.0, float(daily_water or 0.0))
            n = max(1, len(candidates))
            feed_each = total_feed / n
            water_each = total_water / n
            for slot in candidates:
                rows.append({
                    "date": str(slot.date()),
                    "time": _schedule_label(slot),
                    "hour": int(slot.hour),
                    "feed_kg": round(feed_each, 2),
                    "water_liters": round(water_each, 2),
                })
            return rows

        tmp = recent_df.copy()
        for slot in candidates:
            xi = build_predict_row(tmp.iloc[-1], slot, tmp)
            xi["hour"] = int(slot.hour)
            xi["day_of_week"] = int(slot.weekday())
            xi["month"] = int(slot.month)

            fv = max(0.0, float(m_feed.predict(xi)[0]))
            wv = max(0.0, float(m_water.predict(xi)[0]))

            rows.append({
                "date": str(slot.date()),
                "time": _schedule_label(slot),
                "hour": int(slot.hour),
                "feed_kg": round(fv, 2),
                "water_liters": round(wv, 2),
            })

            new_row = pd.DataFrame([{
                "date": slot,
                "feed_kg": fv,
                "water_liters": wv,
                "system": int(last_row.get("system", 1)),
                "day_of_week": int(slot.weekday()),
                "month": int(slot.month),
                "hour": int(slot.hour),
                "lag1_feed": float(tmp.iloc[-1]["feed_kg"]),
                "lag1_water": float(tmp.iloc[-1]["water_liters"]),
                "roll3_feed": float(tmp["feed_kg"].tail(3).mean()),
                "flow": 0.0,
                "level": "0%",
            }])
            tmp = pd.concat([tmp, new_row], ignore_index=True)

    except Exception as e:
        print(f"[ML] schedule prediction skipped: {e}")

    return rows

def render_chart_b64(df: pd.DataFrame, forecast_rows: list = None) -> str:
    """
    Render feed vs water chart as base64 PNG.
    Cloud server generates this; APK only displays it.
    """
    try:
        if df.empty:
            return ""

        plot_df = df.tail(200).copy()

        fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
        ax.set_facecolor(BG)

        ax.plot(
            plot_df["date"],
            plot_df["feed_kg"],
            color=CFEED,
            lw=2.0,
            marker="o",
            ms=3,
            label="Feed / Weight",
        )

        ax.plot(
            plot_df["date"],
            plot_df["water_liters"],
            color=CWATER,
            lw=2.0,
            marker="o",
            ms=3,
            label="Water (L)",
        )

        if forecast_rows:
            try:
                fdf = pd.DataFrame(forecast_rows)
                fdf["date"] = pd.to_datetime(fdf["date"])

                ax.axvline(
                    x=plot_df["date"].iloc[-1],
                    color=BORDER,
                    lw=1,
                    ls="--",
                    alpha=0.5,
                )

                ax.plot(
                    fdf["date"],
                    fdf["feed_kg"],
                    color=CFEED,
                    lw=1.8,
                    ls="--",
                    alpha=0.8,
                    label="Feed Forecast",
                )

                ax.plot(
                    fdf["date"],
                    fdf["water_liters"],
                    color=CWATER,
                    lw=1.8,
                    ls="--",
                    alpha=0.8,
                    label="Water Forecast",
                )
            except Exception as e:
                print(f"[CHART] forecast skipped: {e}")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=25, color=MUTED, fontsize=8, ha="right")
        plt.yticks(color=MUTED, fontsize=9)

        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color=BORDER, lw=0.6, ls="--", alpha=0.7)

        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
            labelcolor=MUTED,
            fontsize=8,
        )

        ax.set_title("Consumption Trends", color=MUTED, fontsize=11, pad=8)

        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            dpi=140,
            facecolor=BG,
        )
        plt.close(fig)

        return base64.b64encode(buf.getvalue()).decode()


    except Exception as e:
        print(f"[CHART] error: {e}")
        return ""
# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def train_once():
    """
    Full training cycle:
      1. Load readings from Firebase
      2. Feature engineering
      3. Train GradientBoosting models (feed + water)
      4. Run ARIMA
      5. Detect anomaly + trend
      6. Forecast 7 days
      7. Write all results back to Firebase
    """
    with _lock:
        if state["training"]:
            return False
        state["training"] = True
        state["status"]   = "training"
        state["error"]    = ""
        state["last_attempt_at"] = datetime.utcnow().isoformat()

    db.write_ml_status("training", state["trained_rows"])

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        HAS_ARIMA = False
        try:
            from statsmodels.tsa.arima.model import ARIMA
            HAS_ARIMA = True
        except Exception:
            pass

        #  1. Load data 
        readings = db.get_readings(limit=ROLLING_WINDOW)
        if not readings:
            db.write_ml_status("waiting", 0, "No data in Firebase yet")
            with _lock:
                state.update(training=False, status="waiting")
            return False

        raw_df = db.readings_to_df(readings)
        daily_df = db.readings_to_daily_df(readings)

        if len(daily_df) >= MIN_DAILY_ROWS:
            df = daily_df
            min_needed = MIN_DAILY_ROWS
            training_source = "daily"
        else:
            df = raw_df
            min_needed = MIN_ROWS
            training_source = "raw"

        if df.empty or len(df) < min_needed:
            msg = (
                f"Need {MIN_DAILY_ROWS} daily rows or {MIN_ROWS} raw rows; "
                f"have {len(daily_df)} daily / {len(raw_df)} raw"
            )
            db.write_ml_status("collecting", len(df), msg)
            with _lock:
                state.update(training=False, status="collecting",
                             trained_rows=len(df),
                             training_source=training_source)
            return False

        total_rows = len(df)
        raw_rows = len(raw_df)
        daily_rows = len(daily_df)
        print(f"[ML] Training on {total_rows} {training_source} rows "
              f"({daily_rows} daily / {raw_rows} raw)…")

        #  2. Feature engineering 
        df = add_features(df)
        X       = df[FEATURES].fillna(0)
        y_feed  = df["feed_kg"]
        y_water = df["water_liters"]

        #  3. Train models
        def make_pipe():
            return Pipeline([
                ("sc", StandardScaler()),
                ("gb", GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=42)),
            ])

        m_feed  = make_pipe()
        m_water = make_pipe()
        m_feed.fit(X, y_feed)
        m_water.fit(X, y_water)

        #  4. Next-day prediction 
        last   = df.iloc[-1]
        nd     = pd.to_datetime(last["date"]) + timedelta(days=1)
        inp    = build_predict_row(last, nd, df)
        feed_v = round(max(0.0, float(m_feed.predict(inp)[0])), 2)
        water_v= round(max(0.0, float(m_water.predict(inp)[0])), 2)

        #  5. ARIMA (stable version)
        arima_feed = arima_water = None

        if HAS_ARIMA and total_rows >= 30:
            try:
                from statsmodels.tsa.arima.model import ARIMA

                af = ARIMA(
                    df["feed_kg"].values,
                    order=(1, 1, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit()

                arima_feed = round(max(0.0, float(af.forecast(1)[0])), 2)

                if df["water_liters"].nunique() <= 2:
                    arima_water = round(max(0.0, float(df["water_liters"].mean())), 2)
                else:
                    aw = ARIMA(
                        df["water_liters"].values,
                        order=(1, 1, 0),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit()

                    arima_water = round(max(0.0, float(aw.forecast(1)[0])), 2)

            except Exception as e:
                print(f"[ML] ARIMA skipped: {e}")
                arima_feed = feed_v
                arima_water = water_v

        else:
            arima_feed = feed_v
            arima_water = water_v

        #  6. Confidence / trend / anomaly 
        conf  = calc_confidence(df, total_rows)
        trend = analyze_trend(df)
        anom  = detect_anomaly(df)

        #  7. 7-day forecast (consistent features) 
        rows_7d = []
        tmp     = df.copy()                
        for _ in range(7):
            l_row = tmp.iloc[-1]
            nd2   = pd.to_datetime(l_row["date"]) + timedelta(days=1)
            xi    = build_predict_row(l_row, nd2, tmp)
            fv    = max(0.0, float(m_feed.predict(xi)[0]))
            wv    = max(0.0, float(m_water.predict(xi)[0]))
            rows_7d.append({
                "date":         str(nd2.date()),
                "feed_kg":      round(fv, 2),
                "water_liters": round(wv, 2),
            })

            new_row = pd.DataFrame([{
                "date":         nd2,
                "feed_kg":      fv,
                "water_liters": wv,
                "system":       1,
                "day_of_week":  nd2.weekday(),
                "month":        nd2.month,
                "hour":         0,
                "lag1_feed":    l_row["feed_kg"],
                "lag1_water":   l_row["water_liters"],
                "roll3_feed":   float(tmp["feed_kg"].tail(3).mean()),
                "flow":         0.0,
                "level":        "0%",
            }])
            tmp = pd.concat([tmp, new_row], ignore_index=True)

        #  8. Scheduled feeding prediction 
        schedule_rows = build_schedule_predictions(last, df, m_feed, m_water, count=4, target_date=nd, daily_feed=feed_v, daily_water=water_v)
        next_sched = schedule_rows[0] if schedule_rows else {}

        #  9. Patterns 
        try:
            pat_sys   = df.groupby("system")[["feed_kg","water_liters"]].mean().to_dict()
            pat_day   = df.groupby("day_of_week")[["feed_kg","water_liters"]].mean().to_dict()
            pat_month = df.groupby("month")[["feed_kg","water_liters"]].mean().to_dict()
        except Exception:
            pat_sys = pat_day = pat_month = {}


        chart_b64 = render_chart_b64(df, rows_7d)
        #  9. Write to Firebase 
        ml_result = {
            "feedKg":     feed_v,
            "waterL":     water_v,
            "predDate":   str(nd.date()),
            "arimaFeed":  arima_feed,
            "arimaWater": arima_water,
            "confidence": conf,
            "confLabel":  confidence_label(conf),
            "trend":      trend["trend"],
            "trendIcon":  trend["icon"],
            "feedDelta":  trend["feedDelta"],
            "waterDelta": trend["waterDelta"],
            "anomaly":    anom["anomaly"],
            "anomalyMsg": anom["message"],
            "modelRows":  total_rows,
            "rawRows":    raw_rows,
            "dailyRows":  daily_rows,
            "trainingSource": training_source,
            "trainedAt":  datetime.utcnow().isoformat(),
            "patSystem":  pat_sys,
            "patDay":     pat_day,
            "patMonth":   pat_month,
            "chartB64":   chart_b64,

            "feedSchedule": schedule_rows,
            "nextFeedTime": next_sched.get("time", ""),
            "nextFeedDate": next_sched.get("date", ""),
            "nextFeedKg": next_sched.get("feed_kg", 0.0),
            "nextFeedWaterL": next_sched.get("water_liters", 0.0),
        }

        ok1 = db.write_ml_result(ml_result)
        ok2 = db.write_forecast_7d(rows_7d)

        if not (ok1 and ok2):
            err = f"Firebase write failed: ml_result={ok1}, forecast_7d={ok2}"
            db.write_ml_status("error", total_rows, err)
            with _lock:
                state.update(
                    trained_rows=total_rows,
                    training=False,
                    status="error",
                    error=err,
                    last_write_ok=False,
                )
            print(f"[ML] {err}")
            return False

        db.write_ml_status("ready", total_rows)

        if anom["anomaly"]:
            alert_key = f"{anom['message']}:{round(float(anom['value']), 2)}"
            alert_now = time.time()
            with _lock:
                last_key = state.get("last_anomaly_key", "")
                last_at = float(state.get("last_anomaly_at", 0.0) or 0.0)
                should_alert = alert_key != last_key or (alert_now - last_at) > 3600
                if should_alert:
                    state["last_anomaly_key"] = alert_key
                    state["last_anomaly_at"] = alert_now
            if should_alert:
                db.push_alert("anomaly", anom["message"], anom["value"])

        with _lock:
            state.update(
                trained_rows=total_rows,
                training=False,
                status="ready",
                error="",
                    last_trained_at=datetime.utcnow().isoformat(),
                    last_write_ok=True,
                    training_source=training_source,
                )

        print(f"[ML] ✅ feed={feed_v}kg water={water_v}L "
              f"conf={int(conf*100)}% trend={trend['trend']} "
              f"firebase_write={'ok' if ok1 and ok2 else 'FAILED'}")
        return True

    except Exception as ex:
        err = str(ex)[:200]
        print(f"[ML] ❌ {err}")
        print(traceback.format_exc())
        db.write_ml_status("error", state.get("trained_rows", 0), err)
        with _lock:
            state.update(training=False, status="error", error=err,
                         last_write_ok=False)
        return False


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def startup_train_once():
    """Run one delayed training attempt after server boot."""
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
                db.write_ml_status("collecting", current,
                                   f"Need {MIN_ROWS} rows, have {current}")
                with _lock:
                    state.update(status="collecting", trained_rows=current)
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
    """Start ML workers once, whether launched by uvicorn import or python file."""
    global _workers_started
    with _lock:
        if _workers_started:
            return False
        _workers_started = True

    threading.Thread(target=training_loop, daemon=True).start()
    threading.Thread(target=startup_train_once, daemon=True).start()
    return True


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI  — Render needs an HTTP server
# ═════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_background_workers()
    yield


app = FastAPI(title="Poultry Farm ML Server", lifespan=lifespan)


@app.get("/")
async def root():
    with _lock:
        s = dict(state)
    return JSONResponse({"service": "Poultry Farm ML", "state": s})


@app.get("/health")
async def health():
    """Ping this every 5min with UptimeRobot to keep Render free tier awake."""
    return JSONResponse({"status": "ok", "ts": datetime.utcnow().isoformat()})


@app.get("/status")
async def get_status():
    with _lock:
        return JSONResponse(dict(state))


@app.post("/retrain")
async def force_retrain():
    """POST to /retrain to manually trigger a training cycle."""
    _event.set()
    return JSONResponse({"message": "Retrain queued"})


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Poultry Farm Cloud ML Server")
    print(f"  Firebase : {db.FIREBASE_URL}")
    print(f"  Retrains : every {RETRAIN_EVERY} rows or {TRAIN_INTERVAL}s")
    print(f"  Min rows : {MIN_ROWS}")
    print("=" * 60)

    # Render injects PORT env var; default 8000 for local testing
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
