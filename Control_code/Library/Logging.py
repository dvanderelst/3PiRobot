import re
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
ORIGIN_STYLE = "\033[1;96m"      # bold bright-cyan
HILITE_NUM = "\033[1;97m"        # bright white for numbers
DIM = "\033[2m"                  # dim for secondary text
SEPARATOR = "─" * 60
ORIGIN_WIDTH = 18                 # width for the [origin] field
LEVEL_WIDTH = 9                   # width for the [LEVEL] field (incl. brackets)

_num_re = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)(ms|us|s|%)?")

def _highlight_numbers(text: str) -> str:
    # Wrap numbers (and their time units) in bright white for scan-ability
    def repl(m):
        val, unit = m.group(1), m.group(2) or ""
        return f"{HILITE_NUM}{val}{unit}{RESET}"
    return _num_re.sub(repl, str(text))

def _format_origin(origin: str) -> str:
    return f"[{origin}]".ljust(ORIGIN_WIDTH)

def _format_level(cat: str) -> str:
    cat_block = f"[{cat}]".ljust(LEVEL_WIDTH)
    return f"{COLORS[cat]}{SYMBOLS[cat]} {cat_block}{RESET}"

def _is_structured(msg) -> bool:
    if isinstance(msg, Mapping):
        return True
    if isinstance(msg, Sequence) and not isinstance(msg, (str, bytes)):
        # list/tuple of pairs or simple values
        return True
    return False

def _render_block(msg) -> list[str]:
    """
    Turn dicts/lists into pretty bullet lists with aligned keys.
    - dict -> key: value lines (keys aligned)
    - list/tuple:
        - if sequence of (key, value) pairs -> align like dict
        - else -> bullets per item
    """
    lines = []

    # Normalize into list of (key, value) or list of values
    if isinstance(msg, Mapping):
        items = list(msg.items())
        keylen = max((len(str(k)) for k, _ in items), default=0)
        for k, v in items:
            k_s = str(k).rjust(keylen)
            v_s = _highlight_numbers(v)
            lines.append(f"   • {k_s}: {v_s}")
        return lines

    if isinstance(msg, Sequence) and not isinstance(msg, (str, bytes)):
        # Detect sequence of pairs
        if all(isinstance(x, Sequence) and len(x) == 2 for x in msg):
            keylen = max((len(str(k)) for k, _ in msg), default=0)
            for k, v in msg:
                k_s = str(k).rjust(keylen)
                v_s = _highlight_numbers(v)
                lines.append(f"   • {k_s}: {v_s}")
        else:
            for x in msg:
                lines.append(f"   • {_highlight_numbers(x)}")
        return lines

    # Fallback (shouldn't reach here)
    lines.append(f"   • {_highlight_numbers(msg)}")
    return lines

def print_message(origin, message, category="INFO"):
    # Verbosity dispatch
    verbose = 2
    if origin == "Controller":
        verbose = controller_verbosity
    if str(origin).startswith("Robot"):
        verbose = client_verbosity

    # Validate level
    category = category.upper()
    if category not in THRESHOLDS:
        raise ValueError(f"Invalid category '{category}'. Must be one of {list(THRESHOLDS.keys())}.")
    if verbose < THRESHOLDS[category]:
        return

    # Left columns (reset first to avoid color bleed)
    origin_col = f"{RESET}{ORIGIN_STYLE}{_format_origin(origin)}{RESET}"
    level_col = _format_level(category)

    # Structured DEBUG blocks for dicts/lists; single-line otherwise
    if category == "DEBUG" and _is_structured(message):
        # Header
        print(f"{origin_col} {level_col}{DIM} Debug details:{RESET}")
        # Separator
        print(f"{origin_col} {DIM}{SEPARATOR}{RESET}")
        # Body
        for line in _render_block(message):
            print(f"{origin_col} {DIM}{line}{RESET}")
        # Footer separator
        print(f"{origin_col} {DIM}{SEPARATOR}{RESET}")
    else:
        # Single-line path
        msg_text = _highlight_numbers(message)
        print(f"{origin_col} {level_col} {msg_text}")
