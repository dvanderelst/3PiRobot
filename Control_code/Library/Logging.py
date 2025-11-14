import re, hashlib
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from Library.Settings import client_verbosity, controller_verbosity, tracker_verbosity

# --- timing flags -------------------------------------------------------------
SHOW_WALLCLOCK    = False  # [HH:MM:SS.mmm]
SHOW_DELTA_GLOBAL = True   # +Δms since any last log
SHOW_DELTA_ORIGIN = True   # o+Δms since last log from same origin
SHOW_SINCE_START  = False   # T+secs since reset/start
# --- alignment options --------------------------------------------------------
FIXED_WIDTH_TIMING = True     # make timing sub-fields fixed width for alignment
ASCII_LEVEL_SYMBOLS = False   # use ASCII level symbols to avoid emoji width issues
# --- config -------------------------------------------------------------------
THRESHOLDS = {"ERROR": 0, "WARNING": 1, "INFO": 2, "DEBUG": 3}
COLORS = {
    "ERROR": "\033[91m",   # red
    "WARNING": "\033[93m", # yellow
    "INFO": "\033[94m",    # blue
    "DEBUG": "\033[92m"    # green
}

SYMBOLS = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️", "DEBUG": "🐞"}
if ASCII_LEVEL_SYMBOLS: SYMBOLS = {"ERROR": "E", "WARNING": "W", "INFO": "I", "DEBUG": "D"}
RESET = "\033[0m"
HILITE_NUM = "\033[1;97m"
DIM = "\033[2m"
SEPARATOR = "─" * 60
ORIGIN_WIDTH = 18
LEVEL_WIDTH = 9

# Toggle this if you prefer basic 8-color origins instead of 256-color
USE_256_ORIGIN = True

# 256-color palette (distinct hues). Change or extend to your taste.
ORIGIN_PALETTE_256 = [33, 39, 69, 75, 111, 141, 171, 201, 207, 214, 220, 190, 154, 118, 82, 46]
# 8/16-color fallback palette
ORIGIN_PALETTE_BASIC = [
    "\033[95m",  # bright magenta
    "\033[92m",  # bright green
    "\033[96m",  # bright cyan
    "\033[91m",  # bright red
    "\033[93m",  # bright yellow
    "\033[94m",  # bright blue
    "\033[97m",  # bright white
]


def set_timing(*, wallclock=None, d_global=None, d_origin=None, since_start=None):
    """Runtime toggle for timing fields. Pass True/False/None (None = leave unchanged)."""
    global SHOW_WALLCLOCK, SHOW_DELTA_GLOBAL, SHOW_DELTA_ORIGIN, SHOW_SINCE_START
    if wallclock is not None:    SHOW_WALLCLOCK = bool(wallclock)
    if d_global is not None:     SHOW_DELTA_GLOBAL = bool(d_global)
    if d_origin is not None:     SHOW_DELTA_ORIGIN = bool(d_origin)
    if since_start is not None:  SHOW_SINCE_START = bool(since_start)

# --- internals ----------------------------------------------------------------
_num_re = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)(ms|us|s|%)?")

# Monotonic reference (process start) and last-event trackers
_T0_NS = time.perf_counter_ns()
_LAST_GLOBAL_NS = _T0_NS
_LAST_BY_ORIGIN_NS = defaultdict(lambda: None)  # origin -> last perf_counter_ns

def reset_timing():
    """Reset the process start and 'last' markers (useful between runs)."""
    global _T0_NS, _LAST_GLOBAL_NS, _LAST_BY_ORIGIN_NS
    _T0_NS = time.perf_counter_ns()
    _LAST_GLOBAL_NS = _T0_NS
    _LAST_BY_ORIGIN_NS.clear()

def _ms(ns: int) -> float:  # ns -> ms float
    return ns / 1e6

def _format_wall(now_s: float) -> str:
    # HH:MM:SS.mmm local time
    lt = time.localtime(now_s)
    ms = int((now_s - int(now_s)) * 1000)
    return f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}.{ms:03d}"

def _timing_prefix(origin: str) -> str:
    """Return '[...timing fields...]' based on enabled flags; '' if all disabled.
       When FIXED_WIDTH_TIMING=True, each field is formatted to a constant width:
       - WALL: 'HH:MM:SS.mmm'        -> width 12
       - GLOB: '+dddd.dms'           -> width 11 (value part 8.1f)
       - ORIG: 'o+dddd.dms'          -> width 11 (value part 7.1f; total matches GLOB)
       - T:    'T+dddd.ddds'         -> width 12 (value part 9.3f)
    """
    if not (SHOW_WALLCLOCK or SHOW_DELTA_GLOBAL or SHOW_DELTA_ORIGIN or SHOW_SINCE_START):
        return ""

    global _LAST_GLOBAL_NS
    now_wall = time.time()
    now_ns = time.perf_counter_ns()

    # deltas
    d_global_ms = _ms(now_ns - _LAST_GLOBAL_NS)
    _LAST_GLOBAL_NS = now_ns

    last_origin_ns = _LAST_BY_ORIGIN_NS[origin]
    d_origin_ms = None if last_origin_ns is None else _ms(now_ns - last_origin_ns)
    _LAST_BY_ORIGIN_NS[origin] = now_ns

    d_since_start_s = (now_ns - _T0_NS) / 1e9

    if not FIXED_WIDTH_TIMING:
        fields = []
        if SHOW_WALLCLOCK:    fields.append(_format_wall(now_wall))                # 12
        if SHOW_DELTA_GLOBAL: fields.append(f"+{d_global_ms:.1f}ms")
        if SHOW_DELTA_ORIGIN: fields.append("o+—" if d_origin_ms is None else f"o+{d_origin_ms:.1f}ms")
        if SHOW_SINCE_START:  fields.append(f"T+{d_since_start_s:.3f}s")
        return "[" + " | ".join(fields) + "]" if fields else ""

    # --- fixed-width formatting ---
    # helpers ensure exact field widths regardless of value size
    def _fw_wall(t):
        # 'HH:MM:SS.mmm' -> 12 chars
        return _format_wall(t)  # already 12

    def _fw_glob(ms):
        # '+dddd.dms' -> total 11 chars (1 + 8 + 2)
        return f"+{ms:8.1f}ms"

    def _fw_orig(ms):
        # 'o+dddd.dms' -> total 11 chars (2 + 7 + 2)
        return f"o+{ms:7.1f}ms" if ms is not None else f"o+{'—':>7}  "

    def _fw_T(sec):
        # 'T+dddd.ddds' -> total 12 chars (2 + 9 + 1)
        return f"T+{sec:9.3f}s"

    fields = []
    if SHOW_WALLCLOCK:    fields.append(_fw_wall(now_wall))     # 12
    if SHOW_DELTA_GLOBAL: fields.append(_fw_glob(d_global_ms))  # 11
    if SHOW_DELTA_ORIGIN: fields.append(_fw_orig(d_origin_ms))  # 11
    if SHOW_SINCE_START:  fields.append(_fw_T(d_since_start_s)) # 12

    return "[" + " | ".join(fields) + "]" if fields else ""


def _highlight_numbers(text: str) -> str:
    def repl(m):
        val, unit = m.group(1), m.group(2) or ""
        return f"{HILITE_NUM}{val}{unit}{RESET}"
    return _num_re.sub(repl, str(text))

def _hash_index(name: str, modulo: int) -> int:
    h = hashlib.md5(name.encode("utf-8")).digest()
    return h[0] % modulo

def _origin_style_for(origin: str) -> str:
    # Special-case controller
    if origin == "Controller":
        return "\033[1;96m"  # bold bright-cyan
    if origin == "Tracker":
        return "\033[1;92m"  # bold bright-green

    # Deterministic color per origin (e.g., Robot01, Robot02, etc.)
    if USE_256_ORIGIN:
        idx = _hash_index(origin, len(ORIGIN_PALETTE_256))
        code = ORIGIN_PALETTE_256[idx]
        return f"\033[1;38;5;{code}m"  # bold 256-color
    else:
        idx = _hash_index(origin, len(ORIGIN_PALETTE_BASIC))
        return ORIGIN_PALETTE_BASIC[idx]

def _format_origin(origin: str) -> str:
    return f"[{origin}]".ljust(ORIGIN_WIDTH)

def _format_level(cat: str) -> str:
    cat_block = f"[{cat}]".ljust(LEVEL_WIDTH)
    return f"{COLORS[cat]}{SYMBOLS[cat]} {cat_block}{RESET}"

def _is_structured(msg) -> bool:
    if isinstance(msg, Mapping): return True
    if isinstance(msg, Sequence) and not isinstance(msg, (str, bytes)): return True
    return False

def _render_block(msg) -> list[str]:
    lines = []
    if isinstance(msg, Mapping):
        items = list(msg.items())
        keylen = max((len(str(k)) for k, _ in items), default=0)
        for k, v in items:
            lines.append(f"   • {str(k).rjust(keylen)}: {_highlight_numbers(v)}")
        return lines
    if isinstance(msg, Sequence) and not isinstance(msg, (str, bytes)):
        if all(isinstance(x, Sequence) and len(x) == 2 for x in msg):
            keylen = max((len(str(k)) for k, _ in msg), default=0)
            for k, v in msg:
                lines.append(f"   • {str(k).rjust(keylen)}: {_highlight_numbers(v)}")
        else:
            for x in msg:
                lines.append(f"   • {_highlight_numbers(x)}")
        return lines
    lines.append(f"   • {_highlight_numbers(msg)}")
    return lines

# --- main ---------------------------------------------------------------------
def print_message(origin, message, category="INFO"):
    # Verbosity
    verbose = 2
    if origin == "Controller": verbose = controller_verbosity
    if str(origin).startswith("Robot"): verbose = client_verbosity
    if origin == 'Tracker': verbose = tracker_verbosity

    # Validate level
    category = category.upper()
    if category not in THRESHOLDS:
        raise ValueError(f"Invalid category '{category}'. Must be one of {list(THRESHOLDS.keys())}.")
    if verbose < THRESHOLDS[category]:
        return

    # Columns
    tcol = _timing_prefix(origin)
    tcol = (tcol + " ") if tcol else ""  # only add a trailing space if we printed timing
    origin_style = _origin_style_for(origin)
    origin_col = f"{RESET}{origin_style}{_format_origin(origin)}{RESET}"
    level_col = _format_level(category)

    # Structured vs single-line
    if category == "DEBUG" and _is_structured(message):
        print(f"{tcol}{origin_col} {level_col}{DIM} Debug details:{RESET}")
        print(f"{tcol}{origin_col} {DIM}{SEPARATOR}{RESET}")
        for line in _render_block(message):
            print(f"{tcol}{origin_col} {DIM}{line}{RESET}")
        print(f"{tcol}{origin_col} {DIM}{SEPARATOR}{RESET}")
    else:
        print(f"{tcol}{origin_col} {level_col} {_highlight_numbers(message)}")
