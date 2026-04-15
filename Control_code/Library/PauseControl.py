import FreeSimpleGUI as sg


class PauseControl:
    def __init__(
        self,
        title="Robot Control",
        min_size=(360, 180),
        font=("Arial", 12),
        keep_on_top=True,
    ):
        self._paused  = False
        self._crashed = False
        self._closed  = False

        sg.set_options(font=font)

        layout = [
            [sg.Text("Data acquisition running", key="-STATUS-", expand_x=True, justification="center")],
            [
                sg.Button("Pause", key="-TOGGLE-", bind_return_key=True, size=(10, 1)),
                sg.Button("Crash", key="-CRASH-",  size=(10, 1)),
            ],
            [sg.Text("Note:"), sg.Input("", key="-NOTE-", expand_x=True)],
        ]

        self._window = sg.Window(
            title,
            layout,
            resizable=True,
            finalize=True,
            keep_on_top=keep_on_top,
            element_justification="center",
        )
        self._window.set_min_size(min_size)
        self._window.size = min_size

    @property
    def paused(self):
        return self._paused

    @property
    def closed(self):
        return self._closed

    def note(self):
        if self._window is None:
            return ""
        values = self._window.read(timeout=0)[1]
        return values.get("-NOTE-", "") if values else ""

    def _set_paused(self, crashed: bool):
        self._paused  = True
        self._crashed = crashed
        status = "CRASH — fix robot, then Resume" if crashed else "Paused — click Resume to continue"
        self._window["-STATUS-"].update(status)
        self._window["-TOGGLE-"].update("Resume")

    def _resume(self):
        self._paused = False
        self._window["-STATUS-"].update("Data acquisition running")
        self._window["-TOGGLE-"].update("Pause")

    def _pump(self, timeout=0):
        if self._closed:
            return
        event, _values = self._window.read(timeout=timeout)
        if event in (sg.WIN_CLOSED, None):
            self._closed = True
            return
        if event == "-TOGGLE-":
            if self._paused:
                self._resume()
            else:
                self._set_paused(crashed=False)
        elif event == "-CRASH-":
            self._set_paused(crashed=True)

    def wait_if_paused(self, poll_ms=100):
        """Wait while paused.  Returns True if resumed from a crash-pause, False otherwise."""
        self._pump(timeout=0)
        if not self._paused:
            return False
        while self._paused and not self._closed:
            self._pump(timeout=poll_ms)
        was_crash = self._crashed
        self._crashed = False
        return was_crash

    def close(self):
        if self._window is not None:
            self._window.close()
            self._window = None


if __name__ == "__main__":
    ctrl = PauseControl()
    try:
        while not ctrl.closed:
            ctrl.wait_if_paused()
            sg.time.sleep(50)
    finally:
        ctrl.close()
