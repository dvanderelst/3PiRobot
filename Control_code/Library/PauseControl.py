import FreeSimpleGUI as sg


class PauseControl:
    def __init__(
        self,
        title="Robot Control",
        min_size=(360, 180),
        font=("Arial", 12),
        keep_on_top=True,
    ):
        self._paused = False
        self._closed = False

        sg.set_options(font=font)

        layout = [
            [sg.Text("Data acquisition running", key="-STATUS-", expand_x=True, justification="center")],
            [
                sg.Button("Pause", key="-TOGGLE-", bind_return_key=True, size=(10, 1)),
                sg.Button("Other", key="-OTHER-", size=(10, 1)),
                sg.Button("Other 2", key="-OTHER2-", size=(10, 1)),
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

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._window["-STATUS-"].update("Paused - click again to resume")
            self._window["-TOGGLE-"].update("Resume")
        else:
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
            self._toggle_pause()

    def wait_if_paused(self, poll_ms=100):
        self._pump(timeout=0)
        while self._paused and not self._closed:
            self._pump(timeout=poll_ms)

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
