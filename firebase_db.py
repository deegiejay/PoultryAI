"""
firebase_db.py  —  Shared Firebase Realtime Database Layer
============================================================
Used by BOTH:
  • cloud_ml_server.py  (writes predictions / alerts)
  • fletapp.py          (reads everything, pure UI)

Pure REST — no Firebase SDK. Works on Android APK, cloud server, desktop.

SETUP (5 min, one-time):
  1. https://console.firebase.google.com → New project
  2. Build → Realtime Database → Create → Start in TEST mode
  3. Copy URL and set FIREBASE_URL below (same in ALL files)

Firebase structure:
  /latest       → overwritten by ESP every 2s        (live display)
  /readings     → append-only log per ESP POST        (history + ML)
  /ml_result    → overwritten by cloud ML             (current prediction)
  /forecast_7d  → overwritten by cloud ML             (7-day table)
  /alerts       → append-only alert log
  /ml_status    → cloud ML heartbeat
"""

import requests
SESSION = requests.Session()
import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

# ══════════════════════════════════════════════════════════════════════════════
# ▶▶  YOUR FIREBASE URL — matches your ESP32 sketch and Render env var
# ══════════════════════════════════════════════════════════════════════════════
FIREBASE_URL = "https://poultry-ai-e901a-default-rtdb.firebaseio.com"
# ══════════════════════════════════════════════════════════════════════════════

TIMEOUT   = 3
CACHE_TTL = 8   # seconds before re-fetching readings

# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE CACHE
# ─────────────────────────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {
    "latest":      None,
    "readings":    [],
    "ml_result":   None,
    "forecast_7d": None,
    "last_fetch":  0,
    "readings_limit": 0,
    "online":      False,
}
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL REST HELPERS
# NOTE: All helpers read FIREBASE_URL at call time so that
#       cloud_ml_server.py can override it via:
#           import firebase_db as db
#           db.FIREBASE_URL = os.getenv("FIREBASE_URL")
#       and the change takes effect immediately.
# ─────────────────────────────────────────────────────────────────────────────

def _base() -> str:
    """Always returns the current module-level FIREBASE_URL."""
    return FIREBASE_URL.rstrip("/")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _dict_values(raw: Any) -> List[Dict]:
    """Return only dict rows from Firebase dict/list payloads."""
    if isinstance(raw, dict):
        return [v for v in raw.values() if isinstance(v, dict)]
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, dict)]
    return []


def _reading_sort_key(rec: Dict) -> float:
    ts = _safe_float(rec.get("timestamp"), 0.0)
    if ts > 1_000_000:
        return ts

    ts_iso = str(rec.get("ts", rec.get("date", "")) or "").strip()
    if not ts_iso:
        return 0.0

    try:
        return datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _get(path: str, params: str = "") -> Optional[Any]:
    """GET from Firebase. Returns parsed JSON or None on error."""
    try:
        url = f"{_base()}/{path}.json{params}"
        r = SESSION.get(url, timeout=TIMEOUT)
        if r.ok:
            with _lock:
                _cache["online"] = True
            return r.json()
    except Exception:
        pass
    with _lock:
        _cache["online"] = False
    return None


def _put(path: str, payload: dict) -> bool:
    """PUT (overwrite) a Firebase node."""
    try:
        url = f"{_base()}/{path}.json"
        r = SESSION.put(url, json=payload, timeout=TIMEOUT)
        with _lock:
            _cache["online"] = r.ok
        if not r.ok:
            print(f"[DB PUT ERROR] {path}: HTTP {r.status_code} {r.text[:160]}")
        return r.ok
    except Exception as e:
        with _lock:
            _cache["online"] = False
        print(f"[DB PUT ERROR] {path}: {e}")
        return False


def _post(path: str, payload: dict) -> bool:
    """POST (append child) to a Firebase node."""
    try:
        url = f"{_base()}/{path}.json"
        r = SESSION.post(url, json=payload, timeout=TIMEOUT)
        with _lock:
            _cache["online"] = r.ok
        if not r.ok:
            print(f"[DB POST ERROR] {path}: HTTP {r.status_code} {r.text[:160]}")
        return r.ok
    except Exception as e:
        with _lock:
            _cache["online"] = False
        print(f"[DB POST ERROR] {path}: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# READ  (APK + cloud server)
# ═════════════════════════════════════════════════════════════════════════════

def get_latest() -> Optional[Dict]:
    """Single fast read of /latest for live dashboard."""
    data = _get("latest")
    if data:
        with _lock:
            _cache["latest"] = data
        return data
    with _lock:
        return _cache.get("latest")


def get_readings(limit: int = 300):
    """
    Fetch last N readings from /readings ordered by timestamp.
    Cached for CACHE_TTL seconds.
    Firebase requires the 'timestamp' index to be defined in rules for
    orderBy queries — if not set yet, results still come back unordered
    and we sort client-side.
    """
    now = time.time()
    with _lock:
        cached = _cache["readings"]
        last_f = _cache["last_fetch"]
        cached_limit = int(_cache.get("readings_limit", 0) or 0)

    # Only use cache if it was fetched with at least the requested limit.
    # This prevents a small mobile query from breaking total count.
    if cached and (now - last_f) < CACHE_TTL and cached_limit >= limit:
        return cached[-limit:] if len(cached) > limit else cached

    # Try ordered query first
    raw = _get("readings", f'?orderBy="timestamp"&limitToLast={limit}')
    if raw is None:
        # Fallback: fetch without ordering (works even without index rules)
        raw = _get("readings")

    readings = _dict_values(raw)
    if readings:
        # Client-side sort by timestamp. Legacy/partial rows are allowed.
        readings.sort(key=_reading_sort_key)
        # Apply limit client-side
        if len(readings) > limit:
            readings = readings[-limit:]
        with _lock:
            _cache["readings"] = readings
            _cache["last_fetch"] = now
            _cache["readings_limit"] = limit
        return readings

    with _lock:
        return _cache.get("readings", [])


def get_reading_count() -> int:
    """Count readings via cached list (Firebase has no COUNT)."""
    raw = _get("readings", "?shallow=true")
    if isinstance(raw, dict):
        return len(raw)
    if isinstance(raw, list):
        return len([v for v in raw if v is not None])
    return len(get_readings(limit=5000))


def get_ml_result() -> Optional[Dict]:
    """
    Fetch latest ML prediction from /ml_result.
    Keys: feedKg, waterL, predDate, confidence, trend, trendIcon,
          anomaly, anomalyMsg, feedDelta, waterDelta,
          arimaFeed, arimaWater, modelRows, trainedAt,
          patSystem, patDay, patMonth
    Falls back to cache on network error.
    """
    data = _get("ml_result")
    if data:
        if "chartB64" not in data:
            data["chartB64"] = ""
        with _lock:
            _cache["ml_result"] = data
        return data
    with _lock:
        return _cache.get("ml_result")


def get_forecast_7d() -> Optional[List[Dict]]:
    """Fetch 7-day forecast list from /forecast_7d."""
    data = _get("forecast_7d")
    if data:
        # Firebase stores as numbered dict {"0":{...},"1":{...}}
        if isinstance(data, dict):
            def sort_key(item):
                key, _row = item
                try:
                    return (0, int(key))
                except Exception:
                    return (1, str(key))

            rows = [row for _key, row in sorted(data.items(), key=sort_key)
                    if isinstance(row, dict)]
        elif isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
        else:
            rows = []

        if rows:
            with _lock:
                _cache["forecast_7d"] = rows
            return rows
    with _lock:
        return _cache.get("forecast_7d")


def get_ml_status() -> Optional[Dict]:
    """Fetch cloud ML heartbeat from /ml_status."""
    return _get("ml_status")


def get_alerts(limit: int = 20) -> List[Dict]:
    """Fetch recent alerts from /alerts."""
    raw = _get("alerts", f'?orderBy="ts"&limitToLast={limit}')
    alerts = _dict_values(raw)
    if alerts:
        alerts.sort(key=lambda x: x.get("ts", ""), reverse=True)
        return alerts
    return []


def get_cache_status() -> Dict:
    with _lock:
        return {
            "online":      _cache["online"],
            "cached_rows": len(_cache["readings"]),
            "has_latest":  _cache["latest"] is not None,
            "has_ml":      _cache["ml_result"] is not None,
            "last_fetch": _cache["last_fetch"],
        }


# ═════════════════════════════════════════════════════════════════════════════
# WRITE  (cloud_ml_server.py only — APK never writes ML data)
# ═════════════════════════════════════════════════════════════════════════════

def write_ml_result(payload: dict) -> bool:
    payload = dict(payload or {})
    payload["writtenAt"] = datetime.utcnow().isoformat()
    return _put("ml_result", payload)


def write_forecast_7d(rows: list) -> bool:
    data = {str(i): row for i, row in enumerate(rows)}
    return _put("forecast_7d", data)


def write_ml_status(status: str, rows: int, error: str = "") -> bool:
    return _put("ml_status", {
        "status":    status,
        "rows":      rows,
        "error":     error,
        "updatedAt": datetime.utcnow().isoformat(),
    })


def push_alert(alert_type: str, message: str, value: float = 0.0) -> bool:
    return _post("alerts", {
        "ts":      datetime.utcnow().isoformat(),
        "type":    alert_type,
        "message": message,
        "value":   value,
    })


def write_device_command(command: str, payload: dict = None,
                         device_id: str = "esp32-poultry-01") -> bool:
    """
    Write a simple command for the ESP32 to poll.

    The app can call this now. The ESP32 sketch must poll
    /commands/{device_id}.json and acknowledge/delete it for the command to
    physically run on the board.
    """
    command = str(command or "").strip().lower()
    device_id = str(device_id or "esp32-poultry-01").strip()
    if not command or not device_id:
        return False

    now = datetime.utcnow().isoformat()
    return _put(f"commands/{device_id}", {
        "id": f"{device_id}-{int(time.time() * 1000)}",
        "command": command,
        "payload": dict(payload or {}),
        "status": "pending",
        "createdAt": now,
        "updatedAt": now,
    })


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSION  (cloud server uses this for ML training)
# ═════════════════════════════════════════════════════════════════════════════

def readings_to_df(readings: List[Dict]):
    """
    Convert Firebase readings list → pandas DataFrame for ML training.

    ESP32 sends:
      timestamp  → Unix epoch seconds (integer from time(nullptr))
      ts         → ISO string "2026-04-20T12:34:56"
      weight     → feed_kg
      totalLiters→ water_liters
      flow, level, dayOfWeek, month

    FIX vs previous version:
      - Parse timestamp as Unix epoch (unit='s') first, fall back to ISO ts
      - Handles both int and float timestamps
    """
    import pandas as pd

    if not readings:
        return pd.DataFrame()

    def _num(value, default=0.0, nonnegative=True):
        try:
            n = float(value)
            if nonnegative and n < 0:
                return 0.0
            return n
        except Exception:
            return default

    rows = []
    for rec in readings:
        try:
            if not isinstance(rec, dict):
                continue

            # Try Unix epoch timestamp first (what ESP sends via time(nullptr))
            ts_val = rec.get("timestamp")
            date = pd.NaT
            try:
                ts_num = float(ts_val)
                if ts_num > 1_000_000:          # looks like epoch
                    date = pd.to_datetime(ts_num, unit="s", errors="coerce")
            except Exception:
                pass

            if pd.isna(date):
                # Fallback: ISO string from "ts" field
                ts_iso = rec.get("ts", "")
                date = pd.to_datetime(ts_iso, errors="coerce") if ts_iso else pd.NaT

            if pd.isna(date):
                date = pd.Timestamp.now()

            # Clamp sensor values to non-negative values before ML training.
            # This prevents impossible readings and predictions like -0.00 kg.
            rows.append({
                "date":         date,
                "feed_kg":      _num(rec.get("weight"), 0.0, True),
                "water_liters": _num(rec.get("totalLiters"), 0.0, True),
                "water_total_liters": _num(rec.get("totalLiters"), 0.0, True),
                "flow":         _num(rec.get("flow"), 0.0, True),
                "level":        str(rec.get("level",          "0%")),
                "day_of_week":  int(_num(rec.get("dayOfWeek"),      0, False)),
                "month":        int(_num(rec.get("month"),          1, False)),
                "system":       int(_num(rec.get("system"), 1, False)),
            })
        except Exception as e:
            print(f"[DB] skip row: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # Strip timezone if present (matplotlib + pandas tz-aware = crash)
    try:
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        pass

    try:
        def _water_delta(series):
            delta = series.astype(float).diff()
            delta = delta.where(delta >= 0, series)
            if len(delta):
                delta.iloc[0] = 0.0
            return delta.clip(lower=0.0)

        df["water_delta_liters"] = (
            df.groupby("system", group_keys=False)["water_total_liters"]
              .apply(_water_delta)
              .reset_index(level=0, drop=True)
        )
    except Exception:
        df["water_delta_liters"] = 0.0

    return df


def readings_to_daily_df(readings: List[Dict]):
    """
    Convert ESP readings to one row per day for ML.

    ESP32 sends totalLiters as a boot-time cumulative counter, so this helper
    sums positive deltas into daily water usage instead of training on the raw
    ever-increasing total.
    """
    import pandas as pd

    raw = readings_to_df(readings)
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame()

    df["day"] = df["date"].dt.floor("D")
    daily = (
        df.groupby(["day", "system"], as_index=False)
          .agg(
              feed_kg=("feed_kg", "mean"),
              water_liters=("water_delta_liters", "sum"),
              flow=("flow", "mean"),
              level=("level", "last"),
          )
          .rename(columns={"day": "date"})
          .sort_values("date")
          .reset_index(drop=True)
    )
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["water_liters"] = daily["water_liters"].clip(lower=0.0)
    daily["feed_kg"] = daily["feed_kg"].clip(lower=0.0)
    return daily

def get_cached_reading_count() -> int:
    """Fast count from currently cached readings. Does not call Firebase."""
    try:
        with _lock:
            return len(_cache.get("readings", []) or [])
    except Exception:
        return 0
