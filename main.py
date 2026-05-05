"""
fletapp.py  —  Poultry Farm Monitoring System  (APK / Browser)
===========================================================
PURE UI — Zero ML on device.
All ML results come from Firebase (written by cloud_ml_server.py).

FLOW:
  ESP32 → Firebase /latest + /readings
  Cloud ML → Firebase /ml_result + /forecast_7d
  This app reads Firebase every 3s → displays live

BUILD APK:
  flet build apk --project "Poultry Farm Monitoring System"

BROWSER MODE (phone on same WiFi):
  python fletapp.py --web

FIXES IN THIS VERSION:
  - ft.Icon.name reassignment uses string literals (e.g. "wifi_off_rounded")
    NOT ft.icons.X objects — that causes AttributeError at runtime
  - ft.icons constants used only for initial construction, strings for updates
  - All ft.Icons / ft.icons references unified to ft.icons (lowercase)
  - firebase_db URL matches ESP32 sketch
"""
import flet as ft
import threading
import time
import traceback
import sys
import warnings
from datetime import datetime

import firebase_db as db

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# ICON STRING CONSTANTS
# Using plain strings for .name reassignment avoids AttributeError.
# ft.icons.X works fine for initial ft.Icon(ft.icons.X) construction,
# but when you do icon_ctrl.name = ft.icons.X it may fail in some Flet
# versions. String literals always work.
# ═════════════════════════════════════════════════════════════════════════════
ICO_CLOUD_OK    = "cloud_done_rounded"
ICO_CLOUD_OFF   = "cloud_off_rounded"
ICO_WIFI_FIND   = "wifi_find_rounded"
ICO_WIFI_OFF    = "wifi_off_rounded"
ICO_CHECK       = "check_circle_outline_rounded"
ICO_WARNING     = "warning_amber_rounded"
ICO_ALERT       = "report_gmailerrorred_rounded"
ICO_STORAGE     = "storage_rounded"
ICO_PSYCH       = "psychology_rounded"
ICO_REFRESH     = "refresh_rounded"
ICO_CALENDAR    = "calendar_month_rounded"

# ═════════════════════════════════════════════════════════════════════════════
# COLORS
# ═════════════════════════════════════════════════════════════════════════════
BG      = "#0e1117"
SURFACE = "#262730"
SURF2   = "#1e2130"
SURF3   = "#161b22"
BORDER  = "#3d4257"
TEXT    = "#fafafa"
MUTED   = "#a0aec0"
BLUE    = "#4da3ff"
GREEN   = "#4ade80"
RED     = "#ff4b4b"
AMBER   = "#f59e0b"
CFEED   = "#50C8FF"
CWATER  = "#1f77b4"

# Phone layout constants
PHONE_MAX_WIDTH = 430
PHONE_SIDE_PAD = 10
PHONE_GAP = 10
PHONE_TOP_SAFE = 40
CHART_PHONE_HEIGHT = 370
CHART_DESKTOP_HEIGHT = 460
LIVE_POLL_SECONDS = 2
DB_REFRESH_SECONDS = 30
ML_REFRESH_SECONDS = 15
DEVICE_ID = "esp32-poultry-01"

# ═════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def divider():
    return ft.Container(height=1, bgcolor=BORDER,
                        margin=ft.margin.symmetric(vertical=18))


def hdr(emoji, title, sub=""):
    # Professional section header. The emoji parameter is kept for compatibility
    # with existing calls, but it is intentionally not displayed.
    items = [ft.Row([
        ft.Text(
            title,
            size=18,
            weight=ft.FontWeight.W_600,
            color=TEXT,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
    ], spacing=8, wrap=False)]
    if sub:
        items.append(ft.Text(sub, size=12, color=MUTED))
    return ft.Column(items, spacing=2, tight=True)


def pill(text, color, bg):
    return ft.Container(
        content=ft.Text(text, size=12, color=color, weight=ft.FontWeight.W_500),
        bgcolor=bg,
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
        border_radius=20,
        border=ft.border.all(1, color + "55"),
    )


def friendly_label(key):
    """Readable table labels for farmers/users."""
    k = str(key).strip()
    labels = {
        "#": "No.",
        "date": "Date / Time",
        "Time": "Date / Time",
        "time": "Feeding Time",
        "kg": "Feed Weight",
        "Feed (kg)": "Feed (kg)",
        "feed_kg": "Feed (kg)",
        "feedKg": "Feed (kg)",
        "L": "Water Used",
        "Water (L)": "Water (L)",
        "water_L": "Water (L)",
        "water_liters": "Water (L)",
        "waterL": "Water (L)",
        "flow": "Water Flow",
        "level": "Water Level",
        "group": "Group",
        "name": "Item",
        "value": "Value",
        "system": "System",
    }
    return labels.get(k, k.replace("_", " ").title())


def _to_number(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _fixed_number(value, decimals=2, clamp_negative=False):
    """Always show friendly fixed digits like 0.00.
    Use clamp_negative=True for readings/targets so users never see -0.00 or negative sensor values.
    """
    try:
        n = float(value)
        if clamp_negative and n < 0:
            n = 0.0
        if abs(n) < 0.005:
            n = 0.0
        return f"{n:.{decimals}f}"
    except Exception:
        return str(value)


def _trim_number(value, decimals=2):
    # Kept for older parts of the UI, but now it also keeps trailing zeros.
    return _fixed_number(value, decimals, clamp_negative=False)


def fmt_kg(value):
    return f"{_fixed_number(value, 2, clamp_negative=True)} kg"


def fmt_liters(value):
    return f"{_fixed_number(value, 2, clamp_negative=True)} L"


def fmt_flow(value):
    return f"{_fixed_number(value, 2, clamp_negative=True)} L/min"


def fmt_percent(value):
    return f"{_fixed_number(value, 2, clamp_negative=True)}%"


def fmt_delta_percent(value):
    """Show change in a simple farmer-friendly way.
    Prevents confusing values like 2515000000.00% when previous data is near zero.
    """
    n = _to_number(value, 0.0)
    try:
        import math
        if math.isnan(n) or math.isinf(n):
            return "not enough data"
    except Exception:
        pass
    if abs(n) < 0.005:
        return "no change"
    word = "increased" if n > 0 else "decreased"
    # Very large percent changes usually happen when the previous average is 0.00.
    # Show a clear warning instead of a long number.
    if abs(n) > 100:
        return f"{word} by over 100.00%"
    return f"{word} by {_fixed_number(abs(n), 2)}%"


def format_duration(seconds):
    """Readable time: seconds → minutes → hours → days."""
    try:
        sec = max(0, int(float(seconds)))
    except Exception:
        sec = 0
    if sec < 60:
        unit = "second" if sec == 1 else "seconds"
        return f"{sec} {unit}"
    if sec < 3600:
        mins = sec // 60
        unit = "minute" if mins == 1 else "minutes"
        return f"{mins} {unit}"
    if sec < 86400:
        hrs = sec // 3600
        unit = "hour" if hrs == 1 else "hours"
        return f"{hrs} {unit}"
    days = sec // 86400
    unit = "day" if days == 1 else "days"
    return f"{days} {unit}"


def friendly_group_value(title, value):
    """Make pattern group names readable instead of raw numeric keys."""
    text = str(value)
    try:
        n = int(float(text))
        if "Weekly" in str(title):
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if 0 <= n <= 6:
                return days[n]
        if "Monthly" in str(title):
            months = ["", "January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
            if 1 <= n <= 12:
                return months[n]
        if "System" in str(title):
            return f"System {n}"
    except Exception:
        pass
    return text


def safe_s(v, key=None):
    """User-friendly value formatting: fewer decimals and clear units."""
    try:
        if v is None or v == "":
            return "--"
        k = str(key or "").lower()
        if isinstance(v, (int, float)):
            if any(x in k for x in ["feed", "kg", "weight"]):
                return fmt_kg(v)
            if any(x in k for x in ["water", "liter", "litre"]):
                return fmt_liters(v)
            if "flow" in k:
                return fmt_flow(v)
            if any(x in k for x in ["percent", "efficiency", "confidence"]):
                return fmt_percent(v)
            return _fixed_number(v, 2, clamp_negative=True)
        return str(v)
    except Exception:
        return str(v)


def safe_float(v, default=0.0):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def parse_user_number(text, default=-1.0):
    """Accept simple user input like 1.20, 1.20 kg, or 2,50 and return a number."""
    try:
        if text is None:
            return default
        t = str(text).strip().lower()
        if not t:
            return default
        for token in ["kg", "liters", "liter", "litres", "litre", "l/min", "l", ","]:
            t = t.replace(token, "." if token == "," else "")
        t = "".join(ch for ch in t if ch.isdigit() or ch in ".-")
        if t in ("", ".", "-", "-."):
            return default
        return float(t)
    except Exception:
        return default


def mk_table_list(rows, idx=False):
    if not rows:
        return ft.DataTable(
            columns=[ft.DataColumn(ft.Text("No data", color=MUTED))],
            rows=[]
        )

    keys = list(rows[0].keys())

    cols = []
    if idx:
        cols.append(ft.DataColumn(ft.Text("#", color=MUTED, size=10)))

    for k in keys:
        cols.append(
            ft.DataColumn(
                ft.Text(friendly_label(k), weight=ft.FontWeight.W_600, color=MUTED, size=12)
            )
        )

    data_rows = []
    for i, row in enumerate(rows):
        cells = []

        if idx:
            cells.append(ft.DataCell(ft.Text(str(i + 1), size=11, color=MUTED)))

        for k in keys:
            cells.append(
                ft.DataCell(ft.Text(safe_s(row.get(k), k), size=11, color=TEXT))
            )

        data_rows.append(
            ft.DataRow(
                cells=cells,
                color={"": SURFACE if i % 2 == 0 else SURF2}
            )
        )

    return ft.DataTable(
        columns=cols,
        rows=data_rows,
        heading_row_color={"": SURF3},
        heading_row_height=32,
        data_row_max_height=32,
        column_spacing=10,
    )


def ins_block(title, emoji, data_dict):
    """Render pattern dict/list from Firebase without pandas."""
    if not data_dict:
        return ft.Container()

    rows = []

    try:
        # CASE 1: {"feed_kg": {"4": 0.2137}, "water_liters": {"4": 0}}
        has_nested_dict = any(isinstance(v, dict) for v in data_dict.values())

        if has_nested_dict:
            keys = set()
            for col_data in data_dict.values():
                if isinstance(col_data, dict):
                    keys.update([str(k) for k in col_data.keys()])

            for key in sorted(keys):
                row = {"group": friendly_group_value(title, key)}
                for col_name, col_data in data_dict.items():
                    if isinstance(col_data, dict):
                        val = col_data.get(key, col_data.get(str(key), "--"))
                        row[col_name] = val
                rows.append(row)

        # CASE 2: {"feed_kg": [None, 0.21367], "water_liters": [None, 0.0]}
        else:
            max_len = 0
            for v in data_dict.values():
                if isinstance(v, list):
                    max_len = max(max_len, len(v))

            if max_len > 0:
                for i in range(max_len):
                    row = {"group": i}
                    for col_name, values in data_dict.items():
                        if isinstance(values, list):
                            val = values[i] if i < len(values) else "--"
                            if val is not None:
                                row[col_name] = val
                    if len(row) > 1:
                        rows.append(row)
            else:
                for k, v in data_dict.items():
                    rows.append({"name": friendly_label(k), "value": v})

    except Exception as e:
        rows = [{"error": str(e)}]

    return ft.Column([
        ft.Row([
            ft.Text(title, size=15, weight=ft.FontWeight.W_600, color=TEXT),
        ], spacing=6),

        ft.Container(
            content=ft.Row(
                [mk_table_list(rows)],
                scroll=ft.ScrollMode.AUTO
            ),
            border_radius=8,
            border=ft.border.all(1, BORDER),
            bgcolor=SURF2,
            padding=6,
        ),
    ], spacing=6)


def conf_bar(value):
    """Confidence bar widget."""
    if value >= 0.80:   label, color = "High",     GREEN
    elif value >= 0.55: label, color = "Medium",   AMBER
    elif value >= 0.30: label, color = "Low",      RED
    else:               label, color = "Very Low", RED
    pct   = int(value * 100)
    full_w = 260
    bar_w = max(4, int(full_w * value))
    return ft.Column([
        ft.Row([
            ft.Text("Prediction confidence:", size=12, color=MUTED),
            ft.Text(f"{pct}% ({label})", size=12, color=color,
                    weight=ft.FontWeight.W_600),
        ], spacing=6),
        ft.Stack([
            ft.Container(height=6, border_radius=4, bgcolor=SURF3, width=full_w),
            ft.Container(height=6, border_radius=4, bgcolor=color, width=bar_w),
        ]),
    ], spacing=4, tight=True)


def make_card_row(items, page_width, cols=4, gap=12):
    """
    Mobile-first metric cards.
    On phones, each card uses full available width to avoid overflow.
    """
    try:
        page_width = int(page_width or 380)
    except Exception:
        page_width = 380

    mobile = page_width < 850

    if mobile:
        cols = 1
        gap = PHONE_GAP
        cw = mobile_width(page_width)
    elif page_width < 1100:
        cols = 2
        cw = max(240, (page_width - 70 - gap * (cols - 1)) // cols)
    else:
        cols = 4
        cw = max(240, (page_width - 70 - gap * (cols - 1)) // cols)

    card_rows, row = [], []
    for label, ctrl in items:
        row.append(ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        label,
                        size=11 if mobile else 12,
                        color=MUTED,
                        max_lines=1,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ctrl,
                ],
                spacing=4 if mobile else 5,
                tight=True,
            ),
            bgcolor=SURFACE,
            padding=ft.padding.symmetric(horizontal=14, vertical=12) if mobile else ft.padding.all(18),
            border_radius=10,
            border=ft.border.all(1, BORDER),
            width=cw,
        ))

        if len(row) == cols:
            card_rows.append(ft.Row(row[:], spacing=gap, wrap=False))
            row = []

    if row:
        card_rows.append(ft.Row(row, spacing=gap, wrap=False))

    return ft.Column(card_rows, spacing=gap)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════
def is_mobile_width(width):
    try:
        return (width or 380) < 850
    except Exception:
        return True

def mobile_width(page_width=None):
    try:
        w = int(page_width or 380)
    except Exception:
        w = 380
    return max(300, min(PHONE_MAX_WIDTH, w - PHONE_SIDE_PAD * 2))


def full_width_container(content, page_width=None, **kwargs):
    return ft.Container(
        content=content,
        width=mobile_width(page_width),
        **kwargs
    )

def main(page: ft.Page):
    page.title      = "Poultry Farm Monitoring System"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor    = BG
    page.padding = ft.padding.only(
        left=PHONE_SIDE_PAD if is_mobile_width(page.width or 380) else 28,
        right=PHONE_SIDE_PAD if is_mobile_width(page.width or 380) else 28,
        top=PHONE_TOP_SAFE if is_mobile_width(page.width or 380) else 24,
        bottom=18,
    )
    # Main page does not scroll; each tab scrolls by itself to avoid blank/overlap scrolling.
    page.scroll = None
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    try:
        page.window.min_width = 380
        page.window.height = 900
    except Exception:
        pass

    S = {"running": True, "no_flow": None, "unread_alerts": 0, "last_alert_key": None}
    update_lock = threading.Lock()

    def su():
        try:
            with update_lock:
                page.update()
        except Exception as e:
            print("PAGE UPDATE ERROR:", e)

    def pw():
        try:
            w = page.width or 0
            if w and w > 0:
                return max(320, min(1200, int(w)))
        except Exception:
            pass

        try:
            w = page.window.width or 0
            if w and w > 0:
                return max(320, min(1200, int(w)))
        except Exception:
            pass

        return 380

    def is_mobile():
        return pw() < 850

    # ══════════════════════════════════════════════════════════════════════════
    # BANNER
    # ══════════════════════════════════════════════════════════════════════════
    # Use ICO_* string constants for icon that gets .name reassigned later
    cl_ic   = ft.Icon(ICO_CLOUD_OK,  color=GREEN, size=16)
    cl_st   = ft.Text("Connecting to Firebase...", size=11 if is_mobile() else 13, color=MUTED, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
    db_lbl  = ft.Text("0 records",              size=11 if is_mobile() else 12, color=MUTED, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
    ml_lbl  = ft.Text("ML: connecting...",   size=10 if is_mobile() else 12, color=MUTED, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
    off_pill = ft.Container(
        content=ft.Text("OFFLINE — cached", size=11, color=AMBER),
        bgcolor="#2a2210",
        padding=ft.padding.symmetric(horizontal=8, vertical=3),
        border_radius=12,
        border=ft.border.all(1, AMBER + "55"),
        visible=False,
    )

    banner = ft.Container(
        width=mobile_width(pw()) if is_mobile() else None,
        content=ft.Column([
            ft.Row([cl_ic, cl_st, ft.Container(expand=True), off_pill], spacing=6),
            ft.Row([
                ft.Icon(ICO_STORAGE, color=MUTED, size=12),
                db_lbl,
                ft.Text(" | ", color=BORDER, size=10),
                ft.Icon(ICO_PSYCH, color=MUTED, size=12),
                ml_lbl,
            ], spacing=4, wrap=False, scroll=ft.ScrollMode.AUTO),
        ], spacing=6),
        bgcolor=SURF2,
        padding=ft.padding.symmetric(
            horizontal=12 if is_mobile() else 18,
            vertical=12 if is_mobile() else 14
        ),
        border_radius=8,
        border=ft.border.all(1, GREEN + "44"),
    )

    startup_note = ft.Container(
        width=mobile_width(pw()) if is_mobile() else None,
        content=ft.Row([
            ft.ProgressRing(width=18, height=18, stroke_width=2, color=BLUE),
            ft.Column([
                ft.Text("Starting dashboard...", size=13, color=TEXT, weight=ft.FontWeight.W_600),
                ft.Text("Loading data", size=11, color=MUTED, max_lines=2),
            ], spacing=1, tight=True)
        ], spacing=10),
        bgcolor=SURF3,
        padding=ft.padding.all(12),
        border_radius=10,
        border=ft.border.all(1, BORDER),
        visible=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — LIVE SENSOR MONITORING
    # ══════════════════════════════════════════════════════════════════════════
    v_wt = ft.Text("--", size=24 if is_mobile() else 28, weight=ft.FontWeight.W_500, color=TEXT)
    v_fl = ft.Text("--", size=24 if is_mobile() else 28, weight=ft.FontWeight.W_500, color=TEXT)
    v_lv = ft.Text("--", size=24 if is_mobile() else 28, weight=ft.FontWeight.W_500, color=TEXT)
    v_tl = ft.Text("--", size=24 if is_mobile() else 28, weight=ft.FontWeight.W_500, color=TEXT)
    v_ls = ft.Text("",   size=10 if is_mobile() else 11, color=MUTED, italic=True, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)

    sensor_ctr = ft.Container()

    def rebuild_sensors():
        sensor_ctr.content = make_card_row([
            ("Feed Weight",  v_wt),
            ("Water Flow",   v_fl),
            ("Water Level",  v_lv),
            ("Total Water",  v_tl),
        ], pw(), cols=4)

    # PREDICTION STATUS CARD
    ai_lbl  = ft.Text("Prediction Status",          size=13, color=MUTED)
    ai_main = ft.Text("Waiting for prediction...",  size=16 if is_mobile() else 18,
                      weight=ft.FontWeight.W_600, color=MUTED, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
    ai_feed = ft.Text("", size=13 if is_mobile() else 14, color=CFEED, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
    ai_watr = ft.Text("", size=13 if is_mobile() else 14, color="#60a5fa", max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
    ai_date = ft.Text("", size=11 if is_mobile() else 12, color=MUTED, italic=True, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
    ai_trnd = ft.Row([], spacing=8, visible=False, wrap=True)
    ai_spin = ft.ProgressBar(color=BLUE, bgcolor=SURF2, value=0, visible=False)

    ai_card = ft.Container(
        width=mobile_width(pw()) if is_mobile() else None,
        content=ft.Column(
            [ai_lbl, ai_main, ai_feed, ai_watr, ai_date, ai_trnd, ai_spin],
            spacing=5, tight=True),
        bgcolor=SURFACE,
        padding=ft.padding.all(20),
        border_radius=8,
        border=ft.border.all(1, BORDER),
    )

    def do_tare(_):
        tare_btn.text = "Sending..."
        tare_btn.disabled = True
        su()

        def _r():
            try:
                ok = db.write_device_command("tare", {"source": "app"}, DEVICE_ID)
            except Exception as ex:
                ok = False
                print("TARE COMMAND ERROR:", ex)

            tare_btn.text = "Tare Command Sent" if ok else "Tare Failed"
            tare_btn.bgcolor = (GREEN + "22") if ok else (RED + "22")
            su()

            time.sleep(3)
            tare_btn.text    = "Tare Scale"
            tare_btn.bgcolor = SURFACE
            tare_btn.disabled = False
            su()

        threading.Thread(target=_r, daemon=True).start()

    tare_btn = ft.ElevatedButton(
        "Tare Scale", bgcolor=SURFACE, color=MUTED, on_click=do_tare,
        style=ft.ButtonStyle(
            side=ft.BorderSide(1, BORDER),
            padding=ft.padding.symmetric(horizontal=14, vertical=9),
        ),
    )

    # Alert box — use ICO_* strings for icons that get .name reassigned
    al_ic = ft.Icon(ICO_WIFI_FIND, color=MUTED, size=18)
    al_mg = ft.Text("Waiting for ESP32 data in Firebase...", size=12 if is_mobile() else 13, color=MUTED, max_lines=3, overflow=ft.TextOverflow.ELLIPSIS)
    al_bx = ft.Container(
        width=mobile_width(pw()) if is_mobile() else None,
        content=ft.Row([al_ic, al_mg], spacing=8),
        bgcolor=SURF2, padding=ft.padding.all(13),
        border_radius=6, border=ft.border.all(1, BORDER),
    )

    live_sec = ft.Column([
        ft.Column([
            hdr("", "Live Sensor Monitoring", ""),
            tare_btn
        ]) if is_mobile() else ft.Row([
            hdr("", "Live Sensor Monitoring", ""),
            ft.Container(expand=True),
            tare_btn,
        ]),
        sensor_ctr,
        v_ls,
    ], spacing=10 if is_mobile() else 14)

    records_alert_sec = ft.Column([
        divider(),
        hdr("", "Alert Status", ""),
        al_bx,
    ], spacing=10 if is_mobile() else 14, key="monitoring_alert_section")

    monitoring_scroll_ref = ft.Ref[ft.Column]()

    # Messenger-style alert notification helper.
    # A small red badge appears on the Monitoring tab when there is a new warning.
    # Opening the Monitoring tab clears the badge because the alert box is already visible there.
    alert_badge_text = ft.Text("1", size=10, color=TEXT, weight=ft.FontWeight.BOLD)
    alert_badge = ft.Container(
        content=alert_badge_text,
        width=18,
        height=18,
        bgcolor=RED,
        border_radius=9,
        alignment=ft.alignment.center,
        visible=False,
    )

    monitoring_tab_label = ft.Container(
        content=ft.Row(
            [
                ft.Text("Monitoring", size=13 if is_mobile() else 15, weight=ft.FontWeight.W_600),
                alert_badge,
            ],
            spacing=6,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
        ),
        padding=ft.padding.symmetric(horizontal=2, vertical=2),
        on_click=lambda e: open_monitoring_alert(),
    )

    def update_monitoring_badge(count=None):
        try:
            n = int(S.get("unread_alerts", 0) if count is None else count)
            if n > 0:
                alert_badge_text.value = str(min(n, 9)) if n <= 9 else "9+"
                alert_badge.visible = True
            else:
                alert_badge.visible = False
        except Exception as e:
            print("BADGE UPDATE ERROR:", e)

    def show_alert_popup(message):
        try:
            # Snackbar works like a light notification: visible but not blocking.
            page.snack_bar = ft.SnackBar(
                content=ft.Row(
                    [
                        ft.Container(width=10, height=10, bgcolor=RED, border_radius=5),
                        ft.Text(message, color=TEXT, size=13, expand=True),
                    ],
                    spacing=10,
                ),
                bgcolor=SURFACE,
                duration=5000,
                show_close_icon=True,
            )
            page.snack_bar.open = True
        except Exception as e:
            print("ALERT POPUP ERROR:", e)

    def mark_alert_unread(alert_key, message):
        try:
            if not alert_key or S.get("last_alert_key") == alert_key:
                return
            S["last_alert_key"] = alert_key
            S["unread_alerts"] = min(int(S.get("unread_alerts", 0) or 0) + 1, 9)
            update_monitoring_badge()
            show_alert_popup(message)
        except Exception as e:
            print("ALERT NOTIFY ERROR:", e)

    def clear_monitoring_alert_badge():
        try:
            S["unread_alerts"] = 0
            update_monitoring_badge(0)
        except Exception as e:
            print("ALERT CLEAR ERROR:", e)

    def open_monitoring_alert():
        """Messenger-style behavior: open Monitoring and scroll to the alert box, then clear badge."""
        try:
            # Ensure Monitoring tab is selected first.
            try:
                objective_tabs.selected_index = 0
            except Exception:
                pass

            # Scroll the Monitoring tab to the alert section.
            col = monitoring_scroll_ref.current
            if col:
                try:
                    col.scroll_to(key="monitoring_alert_section", duration=650, curve=ft.AnimationCurve.EASE_IN_OUT)
                except Exception:
                    # Fallback for Flet versions that do not support keyed scroll here.
                    col.scroll_to(offset=-1, duration=650, curve=ft.AnimationCurve.EASE_IN_OUT)

            clear_monitoring_alert_badge()
            su()
        except Exception as e:
            print("OPEN MONITORING ALERT ERROR:", e)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — TITLE
    # ══════════════════════════════════════════════════════════════════════════
    title_sec = ft.Column([
        divider(),
        ft.Row([
            ft.Column([
                ft.Text(
                    "Poultry Farm Monitoring System",
                    size=17 if is_mobile() else 24,
                    weight=ft.FontWeight.BOLD,
                    color=TEXT,
                    max_lines=2,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.Text("Mobile monitoring dashboard",
                        size=11 if is_mobile() else 12, color=MUTED),
            ], spacing=2, tight=True),
        ], spacing=12),
    ], spacing=0)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — FIREBASE DATABASE RECORDS
    # ══════════════════════════════════════════════════════════════════════════
    db_col  = ft.Column(scroll=ft.ScrollMode.AUTO)
    db_col.controls.append(ft.Text("Loading records...", color=MUTED, size=12))
    db_cnt2 = ft.Text("0 total records", size=12 if is_mobile() else 13, color=MUTED)

    def refresh_db():
        try:
            readings = db.get_readings(limit=100)
        except Exception as e:
            readings = []
            print("REFRESH DB ERROR:", e)

        db_col.controls.clear()

        if not readings:
            db_col.controls.append(
                ft.Container(
                    content=ft.Text("No records yet.", color=MUTED, size=12),
                    padding=ft.padding.all(8),
                )
            )
        else:
            rows = []
            display_rows = readings[-20:] if is_mobile() else readings[-100:]
            for r in display_rows:
                rows.append({
                    "date": str(r.get("ts", ""))[:16],
                    "feed_kg": safe_float(r.get("weight", 0)),
                    "water_liters": safe_float(r.get("totalLiters", 0)),
                    "flow": safe_float(r.get("flow", 0)),
                    "level": str(r.get("level", "--")),
                })

            db_col.controls.append(
                ft.Row(
                    [mk_table_list(rows, idx=True)],
                    scroll=ft.ScrollMode.AUTO,
                )
            )

        try:
            db_cnt2.value = f"{db.get_reading_count():,} total records"
        except Exception:
            db_cnt2.value = "0 total records"
        try:
            su()
        except Exception as e:
            print("REFRESH DB UPDATE ERROR:", e)

    db_ref = ft.IconButton(
        ICO_REFRESH, icon_color=MUTED, tooltip="Refresh",
        on_click=lambda _: threading.Thread(target=refresh_db, daemon=True).start(),
    )

    db_sec = ft.Column([
        divider(),
        ft.Column([
            hdr("🗄️", "Firebase Records", ""),

            ft.Row([
                db_cnt2,
                ft.Container(expand=True),
                db_ref,
            ])
        ], spacing=8),

        ft.Container(
            content=db_col,
            width=mobile_width(pw()) if is_mobile() else None,
            height=CHART_PHONE_HEIGHT if is_mobile() else CHART_DESKTOP_HEIGHT,
            border_radius=8,
            border=ft.border.all(1, BORDER),
            bgcolor=SURF2,
            padding=8,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        ),
    ], spacing=10)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — ARIMA FORECAST (from cloud ML)
    # ══════════════════════════════════════════════════════════════════════════
    ar_fv     = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    ar_wv     = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    ar_info   = ft.Text("",   size=12, color=MUTED)
    arima_ctr = ft.Container()

    def rebuild_arima():
        arima_ctr.content = make_card_row([
            ("Feed Forecast",  ar_fv),
            ("Water Forecast", ar_wv),
        ], pw(), cols=2)

    arima_sec = ft.Column([
        divider(),
        hdr("", "ARIMA Forecast", ""),
        arima_ctr,
        ar_info,
    ], spacing=12, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — ML PREDICTIONS (from cloud ML via Firebase)
    # ══════════════════════════════════════════════════════════════════════════
    ai_fv    = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    ai_wv    = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    pd_txt   = ft.Text("Date: N/A", color=BLUE, size=13)
    ai_st    = ft.Text("Loading prediction...", size=13, color=MUTED)
    ai_sb    = ft.Container(
        content=ai_st, bgcolor=SURF2,
        padding=ft.padding.all(13), border_radius=6,
        border=ft.border.all(1, BORDER),
    )
    pred_ctr = ft.Container()
    conf_col = ft.Column([], visible=False)

    sched_title = ft.Text(
        "Scheduled Feeding Times",
        size=13 if is_mobile() else 14,
        color=MUTED,
        weight=ft.FontWeight.W_600,
    )
    sched_main = ft.Text("--", size=22 if is_mobile() else 26, weight=ft.FontWeight.W_600, color=GREEN)
    sched_sub = ft.Text("", size=11 if is_mobile() else 12, color=MUTED, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
    sched_table_wrap = ft.Container(visible=False)

    sched_card = ft.Container(
        width=mobile_width(pw()) if is_mobile() else None,
        content=ft.Column([
            ft.Row([ft.Icon(ICO_CALENDAR, size=15, color=GREEN), sched_title], spacing=6),
            sched_main,
            sched_sub,
            sched_table_wrap,
        ], spacing=6, tight=True),
        bgcolor=SURFACE,
        padding=ft.padding.all(12 if is_mobile() else 16),
        border_radius=10,
        border=ft.border.all(1, GREEN + "44"),
        visible=False,
    )

    pd_bx    = ft.Container(
        content=ft.Row([ft.Icon(ICO_CALENDAR, size=15, color=BLUE), pd_txt], spacing=8),
        bgcolor="#162133", padding=ft.padding.all(13), border_radius=6,
        border=ft.border.all(1, BLUE + "44"), visible=False,
    )

    def rebuild_pred():
        pred_ctr.content = make_card_row([
            ("Daily Feed Target",  ai_fv),
            ("Daily Water Target", ai_wv),
        ], pw(), cols=2)

    ai_pred_sec = ft.Column([
        divider(),
        hdr(
            "",
            "ML Daily Prediction",
            ""
        ),
        ai_sb, pred_ctr, pd_bx, sched_card, conf_col,
    ], spacing=13, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — FARM INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    ins_col = ft.Column(spacing=20)
    ins_sec = ft.Column([
        divider(),
        hdr("", "Farm Insights", ""),
        ins_col,
    ], spacing=13, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — 7-DAY FORECAST TABLE
    # ══════════════════════════════════════════════════════════════════════════
    fc7_wrap = ft.Container(
        visible=False, border_radius=8,
        border=ft.border.all(1, BORDER), bgcolor=SURF2, padding=8,
    )
    fc7_sec = ft.Column([
        divider(),
        hdr(
            "",
            "7-Day Forecast",
            ""
        ),
        fc7_wrap,
    ], spacing=13, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — CONSUMPTION TRENDS CHART
    # ══════════════════════════════════════════════════════════════════════════
    chart_img = ft.Image(
        src_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0d8AAAAASUVORK5CYII=",
        fit="contain",
        border_radius=8,
        height=CHART_PHONE_HEIGHT if is_mobile() else CHART_DESKTOP_HEIGHT,
        width=mobile_width(pw()) - 8 if is_mobile() else None,
    )

    chart_cap = ft.Text(
        "",
        size=10 if is_mobile() else 11,
        color=MUTED,
        italic=True,
        max_lines=3 if is_mobile() else 1,
        overflow=ft.TextOverflow.ELLIPSIS,
    )

    chart_sec = ft.Column([
        divider(),

        ft.Row([
            ft.Text("", size=18 if is_mobile() else 20),

            ft.Column([
                ft.Text(
                    "Consumption Trends",
                    size=18 if is_mobile() else 20,
                    weight=ft.FontWeight.W_600,
                    color=TEXT,
                ),
                ft.Text(
                    "Feed and water trend",
                    size=11 if is_mobile() else 12,
                    color=MUTED,
                ),
            ], spacing=0, tight=True),

        ], spacing=8),

        chart_cap,

        ft.Container(
            width=mobile_width(pw()) if is_mobile() else None,
            content=ft.Column(
                [
                    chart_img,
                    ft.Text(
                        "",
                        size=10,
                        color=MUTED,
                        visible=False,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=SURF2,
            border_radius=10,
            border=ft.border.all(1, BORDER),
            padding=ft.padding.symmetric(
                horizontal=4 if is_mobile() else 12,
                vertical=8 if is_mobile() else 12,
            ),
            alignment=ft.alignment.center,
        ),

    ], spacing=8 if is_mobile() else 10, visible=False)


    # ══════════════════════════════════════════════════════════════════════════
    # COMPARE — WHOLE-DAY MANUAL VS ML TARGET
    # Added only as an extra tab/section. Existing sections and functions are unchanged.
    # ══════════════════════════════════════════════════════════════════════════
    manual_feed_tf = ft.TextField(
        label="Manual Feed Used Today (kg)",
        hint_text="Example: 1.20",
        keyboard_type=ft.KeyboardType.NUMBER,
        bgcolor=SURF2,
        border_color=BORDER,
        color=TEXT,
    )
    manual_water_tf = ft.TextField(
        label="Manual Water Used Today (L)",
        hint_text="Example: 2.50",
        keyboard_type=ft.KeyboardType.NUMBER,
        bgcolor=SURF2,
        border_color=BORDER,
        color=TEXT,
    )
    cmp_feed_saved = ft.Text("--", size=24, weight=ft.FontWeight.W_500, color=GREEN)
    cmp_water_saved = ft.Text("--", size=24, weight=ft.FontWeight.W_500, color="#60a5fa")
    cmp_efficiency = ft.Text("--", size=24, weight=ft.FontWeight.W_500, color=GREEN)
    cmp_note = ft.Text(
        "Enter today’s manual feed and water used.",
        color=MUTED,
        size=12,
        max_lines=4,
        overflow=ft.TextOverflow.ELLIPSIS,
    )
    cmp_graph_holder = ft.Container(
        width=mobile_width(pw()) if is_mobile() else None,
        content=ft.Column([
            ft.Text("Efficiency Graph", color=TEXT, size=16, weight=ft.FontWeight.W_600),
            ft.Text("Enter values first.", color=MUTED, size=12),
        ], spacing=6),
        bgcolor=SURF2,
        border=ft.border.all(1, BORDER),
        border_radius=10,
        padding=ft.padding.all(14),
    )

    def daily_ml_target():
        """Return whole-day ML feed/water target and source label."""
        try:
            ml = db.get_ml_result() or {}
        except Exception:
            ml = {}
        try:
            forecast = db.get_forecast_7d() or []
        except Exception:
            forecast = []

        today = datetime.now().date().isoformat()

        # Main basis: ML prediction is already a one-day target.
        f = max(0.0, safe_float(ml.get("feedKg", 0), 0))
        w = max(0.0, safe_float(ml.get("waterL", 0), 0))
        if f > 0 or w > 0:
            return f, w, "ML daily prediction"

        # Fallback: today's 7-day forecast.
        for row in forecast or []:
            row_date = str(row.get("date", row.get("predDate", "")))[:10]
            if row_date == today:
                f_total = max(0.0, safe_float(row.get("feed_kg", row.get("feedKg", 0)), 0))
                w_total = max(0.0, safe_float(row.get("water_liters", row.get("waterL", 0)), 0))
                if f_total > 0 or w_total > 0:
                    return f_total, w_total, "today's 7-day ML forecast"

        # Last fallback: sum available scheduled rows only when no daily ML target exists.
        schedule = ml.get("feedSchedule", []) or []
        sched_rows = schedule[:4]
        if sched_rows:
            f_total = sum(max(0.0, safe_float(r.get("feed_kg", r.get("feedKg", 0)), 0)) for r in sched_rows)
            w_total = sum(max(0.0, safe_float(r.get("water_liters", r.get("water_L", r.get("waterL", 0))), 0)) for r in sched_rows)
            if f_total > 0 or w_total > 0:
                return f_total, w_total, "scheduled ML times"

        return 0.0, 0.0, "no ML daily target yet"


    def comparison_bar(label, value, max_value, color, unit, max_width=300):
        value = max(0.0, float(value or 0.0))
        max_value = max(0.001, float(max_value or 0.001))
        bar_width = max(6, min(max_width, int((value / max_value) * max_width)))
        return ft.Column([
            ft.Row([
                ft.Text(label, color=TEXT, size=12, weight=ft.FontWeight.W_600),
                ft.Container(expand=True),
                ft.Text(fmt_kg(value) if unit == "kg" else fmt_liters(value) if unit == "L" else safe_s(value), color=MUTED, size=12),
            ], spacing=6),
            ft.Stack([
                ft.Container(width=max_width, height=16, bgcolor=SURF3, border_radius=10),
                ft.Container(width=bar_width, height=16, bgcolor=color, border_radius=10),
            ]),
        ], spacing=5)

    def make_efficiency_graph(manual_feed, ml_feed, manual_water, ml_water, saved, wsaved, feed_eff, water_eff, source_label):
        bar_w = max(220, min(320, (mobile_width(pw()) - 55) if is_mobile() else 520))
        valid_eff = [max(0.0, e) for e in [feed_eff, water_eff] if e is not None]
        avg_eff = sum(valid_eff) / len(valid_eff) if valid_eff else None
        avg_width = 0 if avg_eff is None else max(6, int(bar_w * max(0.0, min(100.0, avg_eff)) / 100.0))

        controls = [
            ft.Text("Daily Efficiency Graph", color=TEXT, size=17, weight=ft.FontWeight.W_600),
            ft.Text(f"Basis: {source_label}", color=MUTED, size=12, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            ft.Container(height=1, bgcolor=BORDER),
        ]

        if manual_feed >= 0:
            feed_max = max(manual_feed, ml_feed, 0.001)
            controls.extend([
                ft.Text("Feed Used Today", color=TEXT, size=14, weight=ft.FontWeight.W_600),
                comparison_bar("Manual feed used", manual_feed, feed_max, RED, "kg", bar_w),
                comparison_bar("ML daily feed target", ml_feed, feed_max, GREEN, "kg", bar_w),
                ft.Text(
                    f"Feed result: {fmt_kg(abs(saved))} " + ("saved" if saved >= 0 else "over the ML target"),
                    color=GREEN if saved >= 0 else RED,
                    size=12,
                ),
            ])

        if manual_water >= 0:
            water_max = max(manual_water, ml_water, 0.001)
            controls.extend([
                ft.Container(height=1, bgcolor=BORDER),
                ft.Text("Water Used Today", color=TEXT, size=14, weight=ft.FontWeight.W_600),
                comparison_bar("Manual water used", manual_water, water_max, RED, "L", bar_w),
                comparison_bar("ML daily water target", ml_water, water_max, "#60a5fa", "L", bar_w),
                ft.Text(
                    f"Water result: {fmt_liters(abs(wsaved))} " + ("saved" if wsaved >= 0 else "over the ML target"),
                    color=GREEN if wsaved >= 0 else RED,
                    size=12,
                ),
            ])

        controls.extend([
            ft.Container(height=1, bgcolor=BORDER),
            ft.Row([
                ft.Text("Average Efficiency", color=TEXT, size=14, weight=ft.FontWeight.W_600),
                ft.Container(expand=True),
                ft.Text("--" if avg_eff is None else fmt_percent(avg_eff), color=GREEN if (avg_eff or 0) > 0 else MUTED, size=14, weight=ft.FontWeight.W_600),
            ]),
            ft.Stack([
                ft.Container(width=bar_w, height=18, bgcolor=SURF3, border_radius=10),
                ft.Container(width=avg_width, height=18, bgcolor=GREEN if (avg_eff or 0) > 0 else MUTED, border_radius=10),
            ]),
            ft.Text("Higher % means more possible savings.", color=MUTED, size=11, max_lines=2),
        ])
        return ft.Column(controls, spacing=8)

    def compute_comparison(_=None):
        ml_feed, ml_water, source_label = daily_ml_target()
        manual_feed = parse_user_number(manual_feed_tf.value, -1)
        manual_water = parse_user_number(manual_water_tf.value, -1)

        if manual_feed < 0 and manual_water < 0:
            cmp_note.value = "Enter today’s manual feed or water first."
            cmp_graph_holder.content = ft.Text("No graph yet. Enter values first.", color=MUTED, size=12)
            cmp_feed_saved.value = "--"
            cmp_water_saved.value = "--"
            cmp_efficiency.value = "--"
            su()
            return

        # Do not accept negative manual values.
        if manual_feed < 0:
            manual_feed = -1
        if manual_water < 0:
            manual_water = -1

        if ml_feed <= 0 and ml_water <= 0:
            cmp_note.value = "ML daily target is not ready yet."
            cmp_graph_holder.content = ft.Text("No ML daily target yet.", color=MUTED, size=12)
            su()
            return

        saved = ml_feed - manual_feed if manual_feed >= 0 else 0.0
        wsaved = ml_water - manual_water if manual_water >= 0 else 0.0

        feed_eff = None
        if manual_feed >= 0:
            if ml_feed > 0:
                feed_eff = max(-100.0, min(100.0, (saved / ml_feed) * 100.0))
            else:
                feed_eff = 0.0 if manual_feed <= 0 else -100.0
            cmp_feed_saved.value = (fmt_kg(abs(saved)) + (" saved" if saved >= 0 else " over target"))
            cmp_feed_saved.color = GREEN if saved >= 0 else RED
        else:
            cmp_feed_saved.value = "--"
            cmp_feed_saved.color = MUTED

        water_eff = None
        if manual_water >= 0:
            if ml_water > 0:
                water_eff = max(-100.0, min(100.0, (wsaved / ml_water) * 100.0))
            else:
                water_eff = 0.0 if manual_water <= 0 else -100.0
            cmp_water_saved.value = (fmt_liters(abs(wsaved)) + (" saved" if wsaved >= 0 else " over target"))
            cmp_water_saved.color = GREEN if wsaved >= 0 else RED
        else:
            cmp_water_saved.value = "--"
            cmp_water_saved.color = MUTED

        valid_eff = [max(0.0, e) for e in [feed_eff, water_eff] if e is not None]
        cmp_efficiency.value = "--" if not valid_eff else fmt_percent(sum(valid_eff) / len(valid_eff))
        cmp_graph_holder.content = make_efficiency_graph(manual_feed, ml_feed, manual_water, ml_water, saved, wsaved, feed_eff, water_eff, source_label)
        cmp_note.value = f"Done. Basis: {source_label}."
        su()

    compare_button = ft.ElevatedButton(
        "Calculate Efficiency",
        on_click=compute_comparison,
        bgcolor=SURFACE,
        color=TEXT,
        style=ft.ButtonStyle(side=ft.BorderSide(1, BORDER)),
    )

    def rebuild_compare_cards():
        cmp_cards.content = make_card_row([
            ("Feed Difference", cmp_feed_saved),
            ("Water Difference", cmp_water_saved),
            ("Efficiency", cmp_efficiency),
        ], pw(), cols=3)

    cmp_cards = ft.Container()
    compare_sec = ft.Column([
        divider(),
        hdr("", "Manual vs ML Comparison", ""),
        ft.Container(
            width=mobile_width(pw()) if is_mobile() else None,
            content=ft.Column([manual_feed_tf, manual_water_tf, compare_button, cmp_note], spacing=10),
            bgcolor=SURFACE,
            padding=ft.padding.all(14),
            border_radius=10,
            border=ft.border.all(1, BORDER),
        ),
        cmp_cards,
        cmp_graph_holder,
    ], spacing=13)

    # ══════════════════════════════════════════════════════════════════════════
    # RESIZE HANDLER
    # ══════════════════════════════════════════════════════════════════════════
    def on_resize(e):
        rebuild_sensors()
        rebuild_arima()
        rebuild_pred()
        rebuild_compare_cards()

        su()

    # page.on_resized = on_resize

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE UPDATE LOOP — polls Firebase every 3s
    # ══════════════════════════════════════════════════════════════════════════
    def live_loop():
        last_db = 0
        last_ml = 0
        cached_ml = None
        cached_ml_stat = None
        cached_fc7 = None
        cached_count_text = "0"
        refreshing_db = False
        STALE = 20             # seconds before ESP marked as lost

        while True:
            if not S["running"]:
                break
            try:
                now = time.time()

                try:
                    latest = db.get_latest()
                except Exception as e:
                    print("LATEST READ ERROR:", e)
                    latest = None

                try:
                    cache = db.get_cache_status()
                except Exception:
                    cache = {"online": False}

                online = cache.get("online", False)

                # ── Banner ────────────────────────────────────────────────────
                startup_note.visible = False
                off_pill.visible = not online
                cl_ic.name  = ICO_CLOUD_OK  if online else ICO_CLOUD_OFF
                cl_ic.color = GREEN         if online else AMBER
                cl_st.value = ("Firebase connected "
                               if online else "Firebase offline")
                cl_st.color = GREEN if online else AMBER

                # ── Sensor values ─────────────────────────────────────────────
                if latest:
                    # Prefer ESP Unix timestamp if valid. If timestamp is invalid/future,
                    # do not mark readings inaccurate; just show "live".
                    ts = latest.get("timestamp", 0)
                    ts_f = safe_float(ts, 0)
                    if ts_f > 1_000_000 and ts_f <= now + 60:
                        age = max(0, now - ts_f)
                    else:
                        age = 0

                    esp = age < STALE
                    flow = safe_float(latest.get("flow", 0))

                    # IMPORTANT:
                    # Firebase /latest keeps the last saved ESP32 value even when ESP is off.
                    # If the timestamp is stale, do not show it as a live reading.
                    if esp:
                        v_wt.value = fmt_kg(safe_float(latest.get('weight', 0)))
                        v_fl.value = fmt_flow(flow)
                        v_lv.value = str(latest.get("level", "--"))
                        v_tl.value = fmt_liters(safe_float(latest.get('totalLiters', 0)))
                        v_ls.value = (f"Updated {format_duration(age)} ago • {cached_count_text} records")
                    else:
                        v_wt.value = "OFFLINE"
                        v_fl.value = "--"
                        v_lv.value = "--"
                        v_tl.value = "--"
                        v_ls.value = (f"ESP offline • last update {format_duration(age)} ago • {cached_count_text} records")

                    if esp:
                        if flow <= 0:
                            if S["no_flow"] is None:
                                S["no_flow"] = now
                            el = now - S["no_flow"]
                            if el >= 60:
                                al_ic.name    = ICO_ALERT
                                al_ic.color   = RED
                                al_mg.value   = (f"No water flow for "
                                                 f"{format_duration(el)}. Check water line/pump.")
                                al_mg.color   = RED
                                al_bx.bgcolor = "#2a1a1a"
                                al_bx.border  = ft.border.all(1, RED + "44")
                                mark_alert_unread("no_water_flow", "No water flow detected. Check the water line or pump.")
                            else:
                                r = 60 - int(el)
                                al_ic.name    = ICO_WARNING
                                al_ic.color   = AMBER
                                al_mg.value   = f"No water flow. Checking in {format_duration(r)}..."
                                al_mg.color   = AMBER
                                al_bx.bgcolor = "#2a2210"
                                al_bx.border  = ft.border.all(1, AMBER + "44")
                        else:
                            S["no_flow"] = None
                            al_ic.name    = ICO_CHECK
                            al_ic.color   = GREEN
                            al_mg.value   = "Water flow is normal."
                            al_mg.color   = GREEN
                            al_bx.bgcolor = "#1a2e1a"
                            al_bx.border  = ft.border.all(1, GREEN + "44")
                            S["last_alert_key"] = None
                    else:
                        S["no_flow"] = None
                        al_ic.name    = ICO_WIFI_OFF
                        al_ic.color   = AMBER
                        al_mg.value   = f"ESP32 signal lost — last update was {format_duration(age)} ago"
                        al_mg.color   = AMBER
                        al_bx.bgcolor = SURF2
                        al_bx.border  = ft.border.all(1, BORDER)
                        mark_alert_unread("esp_offline", "ESP32 signal lost. Check the device power or Wi-Fi connection.")
                else:
                    v_wt.value = v_fl.value = v_lv.value = v_tl.value = "--"
                    S["no_flow"] = None
                    al_ic.name    = ICO_WIFI_FIND
                    al_ic.color   = MUTED
                    al_mg.value   = "Waiting for ESP32 data..."
                    al_mg.color   = MUTED
                    al_bx.bgcolor = SURF2
                    al_bx.border  = ft.border.all(1, BORDER)

                # ── Cloud ML result ───────────────────────────────────────────
                # ML/cloud data is slower-changing. Do not read it every live sensor tick.
                if now - last_ml > ML_REFRESH_SECONDS:
                    last_ml = now
                    try:
                        cached_ml = db.get_ml_result()
                    except Exception as e:
                        print("ML RESULT ERROR:", e)
                        cached_ml = cached_ml

                    try:
                        cached_ml_stat = db.get_ml_status()
                    except Exception as e:
                        print("ML STATUS ERROR:", e)
                        cached_ml_stat = cached_ml_stat

                ml = cached_ml
                ml_stat = cached_ml_stat

                # ML status banner label
                if ml_stat:
                    s  = ml_stat.get("status", "--")
                    nr = ml_stat.get("rows",    0)
                    ml_lbl.value = {
                        "ready":      f"ML: {nr:,} records trained",
                        "training":   f"ML: retraining {nr:,} records...",
                        "collecting": f"ML: collecting ({nr} records)...",
                        "error":      f"ML: {ml_stat.get('error','')[:40]}",
                        "waiting":    "ML: waiting for data...",
                    }.get(s, f"ML: {s}")

                if ml:
                    pf   = max(0.0, float(ml.get("feedKg",    0.0)))
                    pw_  = max(0.0, float(ml.get("waterL",    0.0)))
                    pd_  = str(ml.get("predDate",    "N/A"))
                    conf = float(ml.get("confidence", 0.0))
                    trend= str(ml.get("trend",       "stable"))
                    tic  = str(ml.get("trendIcon",   ""))
                    anom = bool(ml.get("anomaly",    False))
                    fd   = float(ml.get("feedDelta",  0.0))
                    wd   = float(ml.get("waterDelta", 0.0))
                    nr   = int(ml.get("modelRows",    0))
                    src  = str(ml.get("trainingSource", "") or "").strip()
                    af   = ml.get("arimaFeed",        None)
                    aw_  = ml.get("arimaWater",       None)
                    ta   = str(ml.get("trainedAt",    ""))
                    feed_schedule = ml.get("feedSchedule", []) or []
                    next_feed_time = str(ml.get("nextFeedTime", ""))
                    next_feed_date = str(ml.get("nextFeedDate", ""))
                    next_feed_kg = safe_float(ml.get("nextFeedKg", 0.0))
                    next_feed_water = safe_float(ml.get("nextFeedWaterL", 0.0))

                    # Prediction card in sensor section
                    ai_spin.visible = False
                    ai_main.value   = "Daily Prediction Ready"
                    ai_main.color   = GREEN
                    ai_feed.value   = f"Feed Target: {fmt_kg(pf)}"
                    ai_watr.value   = f"Water Target: {fmt_liters(pw_)}"
                    ai_date.value   = (f"Date: {pd_} • Confidence {fmt_percent(conf*100)}")

                    tc = {"stable":GREEN,"increasing":AMBER,
                          "decreasing":BLUE,"warning":RED}.get(trend, MUTED)
                    tb = {"stable":"#1a2e1a","increasing":"#2a2210",
                          "decreasing":"#162133","warning":"#2a1a1a"}.get(trend, SURF2)
                    ai_trnd.controls = [pill(f"{trend.capitalize()}", tc, tb)]
                    if abs(fd) >= 0.005:
                        ai_trnd.controls.append(
                            ft.Text(f"Feed change: {fmt_delta_percent(fd)}",
                                    size=12, color=AMBER if abs(fd) > 5 else MUTED))
                    if abs(wd) >= 0.005:
                        ai_trnd.controls.append(
                            ft.Text(f"Water change: {fmt_delta_percent(wd)}",
                                    size=12, color=AMBER if abs(wd) > 5 else MUTED))
                    if anom:
                        ai_trnd.controls.append(pill("Anomaly", RED, "#2a1a1a"))
                    ai_trnd.visible = True

                    # ARIMA section
                    if af is not None:
                        ar_fv.value   = fmt_kg(max(0.0, float(af)))
                        ar_wv.value   = fmt_liters(max(0.0, float(aw_)))
                        ar_info.value = (f"Updated: {ta[:19]}")
                        rebuild_arima()
                        arima_sec.visible = True

                    # ML predictions section
                    ai_fv.value  = fmt_kg(pf)
                    ai_wv.value  = fmt_liters(pw_)
                    pd_txt.value = f"Date: {pd_}"
                    pd_bx.visible = True
                    conf_col.controls = [conf_bar(conf)]
                    conf_col.visible  = True
                    src_label = f" {src}" if src else ""
                    ai_st.value   = (f"Ready • {nr:,}{src_label} records")
                    ai_st.color   = GREEN
                    ai_sb.bgcolor = "#1a2e1a"
                    ai_sb.border  = ft.border.all(1, GREEN + "44")
                    rebuild_pred()

                    # Scheduled feed prediction section
                    # ML prediction is a one-day target. The schedule must always
                    # show the full feeding schedule inside that prediction day:
                    # 3:00 AM, 8:00 AM, 11:00 AM, and 2:00 PM.
                    # If Firebase still has an old/partial feedSchedule, the APK
                    # fills the missing times by splitting the daily ML target.
                    try:
                        pred_day = str(pd_)[:10]
                        fixed_schedule = [
                            (3,  "3:00 AM"),
                            (8,  "8:00 AM"),
                            (11, "11:00 AM"),
                            (14, "2:00 PM"),
                        ]

                        # The schedule is the one-day ML target divided by the
                        # four daily feeding times. This keeps the total clear and
                        # avoids partial/old schedule rows from Firebase.
                        default_feed = max(0.0, pf) / 4
                        default_water = max(0.0, pw_) / 4

                        sched_rows = []
                        for hr, label in fixed_schedule:
                            sched_rows.append({
                                "time": label,
                                "feed_kg": default_feed,
                                "water_L": default_water,
                            })

                        sched_main.value = "Daily Schedule"
                        sched_sub.value = (
                            f"Date: {pred_day} • Total: {fmt_kg(pf)} feed, {fmt_liters(pw_)} water"
                        )
                        sched_table_wrap.content = ft.Row([mk_table_list(sched_rows)], scroll=ft.ScrollMode.AUTO)
                        sched_table_wrap.visible = True
                        sched_card.visible = True
                    except Exception:
                        sched_table_wrap.visible = False
                        sched_card.visible = False

                    ai_pred_sec.visible = True

                    # Farm insights
                    pat_sys   = ml.get("patSystem",  {})
                    pat_day   = ml.get("patDay",     {})
                    pat_month = ml.get("patMonth",   {})
                    ins_col.controls.clear()
                    if pat_sys:
                        ins_col.controls.append(
                            ins_block("System Behavior", "", pat_sys))
                    if pat_day:
                        ins_col.controls.append(
                            ins_block("Weekly Pattern", "", pat_day))
                    if pat_month:
                        ins_col.controls.append(
                            ins_block("Monthly Pattern", "", pat_month))
                    if ins_col.controls:
                        ins_sec.visible = True

                else:
                    # No ML result yet
                    ai_spin.visible = (ml_stat or {}).get("status") == "training"
                    ai_main.value   = "Waiting for ML predictions..."
                    ai_main.color   = MUTED
                    ai_feed.value = ai_watr.value = ai_date.value = ""
                    ai_trnd.visible = False
                    try:
                        sched_card.visible = False
                    except Exception:
                        pass

                # Banner DB count
                try:
                    db_lbl.value = f"{db.get_reading_count():,} records"
                except Exception:
                    db_lbl.value = "0 records"

                # 7-day forecast
                if now - last_ml < 1:
                    try:
                        cached_fc7 = db.get_forecast_7d()
                    except Exception as e:
                        print("FORECAST READ ERROR:", e)

                fc7_raw = cached_fc7
                if fc7_raw:
                    try:
                        fc7_wrap.content = mk_table_list(fc7_raw)
                        fc7_wrap.visible = True
                        fc7_sec.visible = True
                    except Exception:
                        pass

                # DB table/count refresh is heavier; keep it separate from live sensor.
                if now - last_db > DB_REFRESH_SECONDS:
                    last_db = now

                    def _refresh_db_bg():
                        nonlocal cached_count_text, refreshing_db
                        if refreshing_db:
                            return
                        refreshing_db = True
                        try:
                            refresh_db()
                            try:
                                cached_count_text = f"{db.get_reading_count():,}"
                            except Exception:
                                cached_count_text = cached_count_text
                        finally:
                            refreshing_db = False

                    threading.Thread(target=_refresh_db_bg, daemon=True).start()

                # Chart image from ML server result only
                if ml:
                    chart_b64 = ml.get("chartB64", "")

                    if chart_b64 and len(chart_b64) > 100:
                        chart_img.src_base64 = chart_b64
                        chart_img.visible = True
                        chart_cap.value = "Updated"
                    else:
                        chart_img.visible = False
                        chart_cap.value = "Waiting for chart..."

                    chart_sec.visible = True

                su()

            except Exception:
                print(traceback.format_exc())

            time.sleep(LIVE_POLL_SECONDS)

    page.on_disconnect = lambda e: S.update(running=False)

    # Initial layout build
    rebuild_sensors()
    rebuild_arima()
    rebuild_pred()
    rebuild_compare_cards()

    def objective_tab_page(*controls, scroll_ref=None):
        # Keep the same content/design, but make each tab handle its own vertical scroll.
        # This prevents the whole app from continuing into blank space on phones.
        return ft.Container(
            content=ft.Column(
                list(controls),
                ref=scroll_ref,
                spacing=8,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
            padding=ft.padding.only(top=8, bottom=24),
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

    monitoring_tab = ft.Tab(
        tab_content=monitoring_tab_label,
        content=objective_tab_page(live_sec, ai_card, records_alert_sec, scroll_ref=monitoring_scroll_ref),
    )

    prediction_tab = ft.Tab(
        text="Prediction",
        content=objective_tab_page(ai_pred_sec, arima_sec, fc7_sec),
    )
    insights_tab = ft.Tab(
        text="Insights",
        content=objective_tab_page(chart_sec, ins_sec),
    )
    compare_tab = ft.Tab(
        text="Compare",
        content=objective_tab_page(compare_sec),
    )
    records_tab = ft.Tab(
        text="Records",
        content=objective_tab_page(db_sec),
    )

    def handle_tab_change(e):
        try:
            idx = int(getattr(e.control, "selected_index", 0) or 0)
            if idx == 0:
                # User has opened Monitoring; jump to the alert box like a notification thread.
                open_monitoring_alert()
        except Exception as ex:
            print("TAB CHANGE ERROR:", ex)

    objective_tabs = ft.Tabs(
        selected_index=0,
        animation_duration=250,
        scrollable=True,
        expand=True,
        on_change=handle_tab_change,
        tabs=[
            monitoring_tab,
            prediction_tab,
            insights_tab,
            compare_tab,
            records_tab,
        ],
    )

    page.add(
        ft.Column(
            [
                banner,
                startup_note,
                ft.Container(height=8),
                title_sec,
                ft.Container(height=6),
                objective_tabs,
            ],
            expand=True,
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )

    threading.Thread(target=live_loop, daemon=True).start()
    threading.Thread(target=refresh_db, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# APK mode:     flet build apk  → native ft.app(target=main), NOT web browser mode
# Browser mode: python fletapp.py --web
# Desktop mode: python fletapp.py
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # sys.argv is safe on desktop/browser but may be empty on APK
    argv = sys.argv if sys.argv else []
    web  = "--web" in argv or "--browser" in argv

    if web:
        try:
            import socket as _s
            _sk = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            _sk.connect(("8.8.8.8", 80))
            _ip = _sk.getsockname()[0]
            _sk.close()
        except Exception:
            _ip = "127.0.0.1"
        PORT = 8550
        print(f"\n🌐 Browser mode — open on phone: http://{_ip}:{PORT}\n")
        ft.app(target=main, view=ft.AppView.WEB_BROWSER,
               port=PORT, host="0.0.0.0")
    else:
        print("📱 Running native app mode")
        ft.app(target=main)
