import FreeSimpleGUI as sg

def ask_yes_no(
    message,
    title="Confirm",
    ask_text=False,
    text_label="Input:",
    default_text="",
    min_size=(420, 200),
    font=("Arial", 14),
    theme=None,
    scaling=1.0,
    keep_on_top=True,
    center="cursor"  # 'cursor' | 'primary' | 'monitor'
):
    """
    Returns: (choice, text)
      - choice is "Yes", "No", or None (window closed/esc)
      - text is the entered string (None if ask_text=False)
    """
    if theme:
        sg.theme(theme)
    sg.set_options(font=font, scaling=scaling)

    # Layout
    rows = [[sg.Text(message, expand_x=True, justification="center")]]
    if ask_text:
        rows += [[sg.Text(text_label, size=(12,1)), sg.Input(default_text, key="-IN-")]]
    rows += [[sg.Push(), sg.Button("Yes", bind_return_key=True), sg.Button("No"), sg.Push()]]

    # Window
    win = sg.Window(
        title,
        rows,
        resizable=True,
        finalize=True,
        keep_on_top=keep_on_top,
        modal=True,
        element_justification="center"
    )

    # Enforce minimum size (and use as starting geometry)
    win.set_min_size(min_size)
    win.size = min_size

    # --- Centering helpers ---
    def _center_primary():
        sw, sh = win.TKroot.winfo_screenwidth(), win.TKroot.winfo_screenheight()
        ww, wh = win.TKroot.winfo_width(), win.TKroot.winfo_height()
        x = (sw - ww) // 2
        y = (sh - wh) // 2
        win.move(x, y)

    def _center_cursor():
        # ensure geometry is computed
        win.TKroot.update_idletasks()
        ww, wh = win.TKroot.winfo_width(), win.TKroot.winfo_height()
        px, py = win.TKroot.winfo_pointerx(), win.TKroot.winfo_pointery()
        sw, sh = win.TKroot.winfo_screenwidth(), win.TKroot.winfo_screenheight()
        # center around pointer, clamped to virtual desktop
        x = max(0, min(px - ww // 2, sw - ww))
        y = max(0, min(py - wh // 2, sh - wh))
        win.move(x, y)

    def _center_monitor_under_cursor():
        try:
            from screeninfo import get_monitors  # mamba install screeninfo
            win.TKroot.update_idletasks()
            ww, wh = win.TKroot.winfo_width(), win.TKroot.winfo_height()
            px, py = win.TKroot.winfo_pointerx(), win.TKroot.winfo_pointery()

            # find monitor containing pointer
            mon = None
            for m in get_monitors():
                if (m.x <= px < m.x + m.width) and (m.y <= py < m.y + m.height):
                    mon = m
                    break
            if mon is None:
                return _center_cursor()  # fallback

            x = mon.x + (m.width - ww) // 2
            y = mon.y + (m.height - wh) // 2
            win.move(x, y)
        except Exception:
            _center_cursor()  # fallback if screeninfo not present/working

    # Perform centering
    if center == "primary":
        _center_primary()
    elif center == "monitor":
        _center_monitor_under_cursor()
    else:
        _center_cursor()

    # Focus input if present
    if ask_text:
        win["-IN-"].set_focus()

    event, values = win.read(close=True)
    text = values.get("-IN-") if (ask_text and values) else None

    if event in (sg.WIN_CLOSED, None):
        return None, text
    return event, text


# -------------------------
# Examples
if __name__ == "__main__":
    # Center on monitor under cursor (best for multi-monitor)
    choice, _ = ask_yes_no("Continue processing?", title="Continue?",
                           theme="DarkBlue3", scaling=1.2, center="monitor")
    print("Choice:", choice)

    # Ask text + center around cursor without extra deps
    choice, name = ask_yes_no("Proceed and record your name?",
                              title="Confirm Action",
                              ask_text=True, text_label="Your name:",
                              default_text="Dieter",
                              theme="LightGreen", scaling=1.3,
                              min_size=(500, 240),
                              center="cursor")
    print("Choice:", choice, "| Name:", name)
