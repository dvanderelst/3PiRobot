
from Library.Settings import controller_verbosity
from Library.Settings import client_verbosity

def print_message(origin, message, category="INFO"):
    # --- configuration --------------------------------------------------
    threshold_levels = {
        "ERROR": 0,
        "WARNING": 1,
        "INFO": 2,
        "DEBUG": 3
    }
    colors = {
        "ERROR": "\033[91m",   # red
        "WARNING": "\033[93m", # yellow
        "INFO": "\033[94m",    # blue
        "DEBUG": "\033[92m"    # green
    }
    origin_style = "\033[1;96m"  # bold bright-cyan
    reset = "\033[0m"

    verbose = 2
    if origin == "Controller":
        verbose = controller_verbosity
    if origin.startswith("Robot"):
        verbose = client_verbosity

    # --- validation -----------------------------------------------------
    category = category.upper()
    if category not in threshold_levels:
        raise ValueError(
            f"Invalid category '{category}'. "
            f"Must be one of {list(threshold_levels.keys())}."
        )

    # --- threshold filter -----------------------------------------------
    if verbose < threshold_levels[category]:
        return

    # --- formatting -----------------------------------------------------
    cat_colour = colors[category]
    origin_fmt = f"[{origin}]".ljust(20)  # pad origin to 20 chars

    print(f"{origin_style}{origin_fmt}{reset} "
          f"{cat_colour}[{category}]{reset} {message}")
