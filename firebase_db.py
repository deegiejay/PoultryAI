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
    "readings_all": False,
    "reading_count": None,
    "count_fetch": 0,
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


def _looks_like_reading(obj: Any) -> bool:
    """Detect a real sensor reading row, not just a parent container."""
    if not isinstance(obj, dict):
        return False
    has_time = any(k in obj for k in ("timestamp", "ts", "date", "createdAt"))
    has_sensor = any(k in obj for k in ("weight", "totalLiters", "flow", "level"))
    return bool(has_time and has_sensor)


def _flatten_readings(raw: Any) -> List[Dict]:
    """
    Return every real reading from Firebase payloads.

    This fixes accidental imports like /readings/readings/... and also prevents
    data from being ignored when Firebase returns nested dictionaries/lists.
    """
    rows: List[Dict] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if _looks_like_reading(obj):
                rows.append(obj)
                return
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(raw)
    return rows


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
    Fetch readings from /readings.

    limit > 0  → fetch latest N rows for fast app views.
    limit <= 0 → fetch ALL available rows for ML training/debug.

    This version removes the old artificial 5,000-count behavior and also
    flattens nested Firebase imports so rows are not wasted or ignored.
    """
    now = time.time()
    fetch_all = limit is None or int(limit or 0) <= 0
    req_limit = 0 if fetch_all else int(limit)

    with _lock:
        cached = list(_cache.get("readings") or [])
        last_f = float(_cache.get("last_fetch") or 0)
        cached_limit = int(_cache.get("readings_limit", 0) or 0)
        cached_all = bool(_cache.get("readings_all", False))

    # Use cache only when it satisfies the request.
    if cached and (now - last_f) < CACHE_TTL:
        if fetch_all and cached_all:
            return cached
        if not fetch_all and (cached_all or cached_limit >= req_limit):
            return cached[-req_limit:] if len(cached) > req_limit else cached

    if fetch_all:
        raw = _get("readings")
    else:
        raw = _get("readings", f'?orderBy="timestamp"&limitToLast={req_limit}')
        if raw is None:
            raw = _get("readings")

    readings = _flatten_readings(raw)
    if readings:
        readings.sort(key=_reading_sort_key)
        if (not fetch_all) and len(readings) > req_limit:
            readings = readings[-req_limit:]

        with _lock:
            _cache["readings"] = readings
            _cache["last_fetch"] = now
            _cache["readings_limit"] = 0 if fetch_all else req_limit
            _cache["readings_all"] = fetch_all
            if fetch_all:
                _cache["reading_count"] = len(readings)
                _cache["count_fetch"] = now
        return readings

    with _lock:
        cached = list(_cache.get("readings") or [])
    if fetch_all:
        return cached
    return cached[-req_limit:] if cached and len(cached) > req_limit else cached


def get_all_readings():
    """Fetch all historical readings for the cloud ML server."""
    return get_readings(limit=0)


def get_reading_count() -> int:
    """
    Count all /readings records without the old 5,000 artificial limit.

    Primary method uses Firebase shallow=true so it counts keys without
    downloading every row. If that fails or looks nested, it falls back to a
    full flattened read.
    """
    now = time.time()
    with _lock:
        cached_count = _cache.get("reading_count")
        count_fetch = float(_cache.get("count_fetch") or 0)
    if cached_count is not None and (now - count_fetch) < CACHE_TTL:
        return int(cached_count)

    raw = _get("readings", "?shallow=true")
    count = None
    if isinstance(raw, dict):
        # Normal import: /readings/<date-key> = row.  This gives the true count.
        count = len(raw)
        # If user accidentally imported root JSON inside /readings, shallow count may be 1.
        if count <= 1 and "readings" in raw:
            count = None
    elif isinstance(raw, list):
        count = len([x for x in raw if x is not None])

    if count is None:
        count = len(get_readings(limit=0))

    with _lock:
        _cache["reading_count"] = int(count)
        _cache["count_fetch"] = now
    return int(count)


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
            "cached_all": bool(_cache.get("readings_all", False)),
            "reading_count": _cache.get("reading_count"),
            "has_latest":  _cache["latest"] is not None,
            "has_ml":      _cache["ml_result"] is not None,
            "last_fetch": _cache["last_fetch"],
        }


# ═════════════════════════════════════════════════════════════════════════════
# WRITE  (cloud_ml_server.py only — APK never writes ML data)
# ═════════════════════════════════════════════════════════════════════════════

def write_ml_result(payload: dict) -> bool:
    payload["writtenAt"] = datetime.utcnow().isoformat()
    return _put("ml_result", payload)


def write_forecast_7d(rows: list) -> bool:
    data = {str(i): row for i, row in enumerate(rows)}
    return _put("forecast_7d", data)


def write_ml_status(status: str, rows: int, error: str = "", **extra) -> bool:
    """Write cloud ML heartbeat/status. Extra keys are allowed for debug UI."""
    payload = {
        "status":    status,
        "rows":      rows,
        "error":     error,
        "updatedAt": datetime.utcnow().isoformat(),
    }
    if extra:
        payload.update(extra)
    return _put("ml_status", payload)


def push_alert(alert_type: str, message: str, value: float = 0.0) -> bool:
    return _post("alerts", {
        "ts":      datetime.utcnow().isoformat(),
        "type":    alert_type,
        "message": message,
        "value":   value,
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
                "flow":         _num(rec.get("flow"), 0.0, True),
                "level":        str(rec.get("level",          "0%")),
                "day_of_week":  int(_num(rec.get("dayOfWeek"),      0, False)),
                "month":        int(_num(rec.get("month"),          1, False)),
                "system":       1,
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

    return df

def get_cached_reading_count() -> int:
    """Fast count from currently cached readings. Does not call Firebase."""
    try:
        with _lock:
            return len(_cache.get("readings", []) or [])
    except Exception:
        return 0
