controller_verbosity = 2  # 0=errors, 1=warnings, 2=info
robot_verbosity = 2  # 0=errors, 1=warnings, 2=info

def print_message(origin, message, category="INFO"):
    # --- configuration --------------------------------------------------
    threshold_levels = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    colors = {"ERROR": "\033[91m", "WARNING": "\033[93m", "INFO": "\033[94m"}
    origin_style = "\033[1;96m"  # bold bright‑cyan
    reset = "\033[0m"

    verbose = 2
    if origin == "Controller": verbose = controller_verbosity
    if origin.startswith("Robot"): verbose = robot_verbosity

    #pad origin to 10 characters


    # assert that category is one of the threshold_levels keys
    category = category.upper()
    if category not in threshold_levels:
        raise ValueError(f"Invalid category '{category}'. Must be one of {list(threshold_levels.keys())}.")

    # --------------------------------------------------------------------
    if verbose < threshold_levels.get(category): return
    cat_colour = colors.get(category)

    origin = f"[{origin}]"
    origin = origin.ljust(20)
    print(f"{origin_style}{origin}{reset} "
          f"{cat_colour}[{category}]{reset} {message}")