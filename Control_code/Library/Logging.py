import re, hashlib
from collections.abc import Mapping, Sequence
from Library.Settings import client_verbosity, controller_verbosity

# --- config -------------------------------------------------------------------
THRESHOLDS = {"ERROR": 0, "WARNING": 1, "INFO": 2, "DEBUG": 3}
COLORS = {
    "ERROR": "\033[91m",   # red
    "WARNING": "\033[93m", # yellow
    "INFO": "\033[94m",    # blue
    "DEBUG": "\033[92m"    # green
}
SYMBOLS = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️", "DEBUG": "🐞"}
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

_num_re = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)(ms|us|s|%)?")

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

def print_message(origin, message, category="INFO"):
    # Verbosity
    verbose = 2
    if origin == "Controller": verbose = controller_verbosity
    if str(origin).startswith("Robot"): verbose = client_verbosity

    # Validate level
    category = category.upper()
    if category not in THRESHOLDS:
        raise ValueError(f"Invalid category '{category}'. Must be one of {list(THRESHOLDS.keys())}.")
    if verbose < THRESHOLDS[category]:
        return

    # Columns
    origin_style = _origin_style_for(origin)
    origin_col = f"{RESET}{origin_style}{_format_origin(origin)}{RESET}"
    level_col = _format_level(category)

    # Structured vs single-line
    if category == "DEBUG" and _is_structured(message):
        print(f"{origin_col} {level_col}{DIM} Debug details:{RESET}")
        print(f"{origin_col} {DIM}{SEPARATOR}{RESET}")
        for line in _render_block(message):
            print(f"{origin_col} {DIM}{line}{RESET}")
        print(f"{origin_col} {DIM}{SEPARATOR}{RESET}")
    else:
        print(f"{origin_col} {level_col} {_highlight_numbers(message)}")
