import threading
import time
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
        self._note    = ""

        self._title       = title
        self._min_size    = min_size
        self._font        = font
        self._keep_on_top = keep_on_top

        # Signal main thread once the window exists and is ready.
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._gui_loop, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _gui_loop(self):
        sg.set_options(font=self._font)
        layout = [
            [sg.Text("Data acquisition running", key="-STATUS-", expand_x=True, justification="center")],
            [
                sg.Button("Pause", key="-TOGGLE-", bind_return_key=True, size=(10, 1)),
                sg.Button("Crash", key="-CRASH-",  size=(10, 1)),
            ],
            [sg.Text("Note:"), sg.Input("", key="-NOTE-", expand_x=True)],
        ]
        window = sg.Window(
            self._title,
            layout,
            resizable=True,
            finalize=True,
            keep_on_top=self._keep_on_top,
            element_justification="center",
        )
        window.set_min_size(self._min_size)
        window.size = self._min_size
        self._ready.set()

        while True:
            event, values = window.read(timeout=100)
            if event in (sg.WIN_CLOSED, None):
                self._closed = True
                break
            if values:
                self._note = values.get("-NOTE-", "")
            if event == "-TOGGLE-":
                if self._paused:
                    self._paused = False
                    window["-STATUS-"].update("Data acquisition running")
                    window["-TOGGLE-"].update("Pause")
                else:
                    self._paused  = True
                    self._crashed = False
                    window["-STATUS-"].update("Paused — click Resume to continue")
                    window["-TOGGLE-"].update("Resume")
            elif event == "-CRASH-":
                self._paused  = True
                self._crashed = True
                window["-STATUS-"].update("CRASH — fix robot, then Resume")
                window["-TOGGLE-"].update("Resume")

        window.close()

    @property
    def paused(self):
        return self._paused

    @property
    def closed(self):
        return self._closed

    def note(self):
        return self._note

    def wait_if_paused(self, poll_ms=100):
        """Block while paused. Returns True if resumed from a crash-pause, False otherwise."""
        if not self._paused:
            return False
        while self._paused and not self._closed:
            time.sleep(poll_ms / 1000.0)
        was_crash = self._crashed
        self._crashed = False
        return was_crash

    def close(self):
        # The GUI thread will clean up its own window; just signal it to stop.
        self._closed = True


if __name__ == "__main__":
    ctrl = PauseControl()
    try:
        while not ctrl.closed:
            ctrl.wait_if_paused()
            time.sleep(0.05)
    finally:
        ctrl.close()
