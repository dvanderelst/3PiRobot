# update_gui.py
import threading, queue, traceback
from pathlib import Path
from tkinter import Tk, StringVar, END, BooleanVar
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

# your modules
from library import tools
from library import rshell
from library import settings

origin_folder = settings.origin_folder
staging_folder = settings.staging_folder

def count_files(d):
    p = Path(d)
    return sum(1 for x in p.rglob("*") if x.is_file())

class ProgressUI:
    def __init__(self, ports):
        self.root = Tk()
        self.root.title("Robot Updater")
        self.ports = ports
        self.total_steps = len(ports) * 3  # will be recalculated per run
        self.done_steps = 0
        self.cancel = False

        # Widgets
        self.status = StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status).pack(padx=12, pady=(12,4), anchor="w")

        self.pbar = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", maximum=self.total_steps)
        self.pbar.pack(fill='both', padx=12, pady=6)

        self.substatus = StringVar(value="")
        ttk.Label(self.root, textvariable=self.substatus, foreground="#555").pack(padx=12, pady=(0,6), anchor="w")

        # Toggles
        self.full_update_var = BooleanVar(value=False)     # include fixed libs + mirror
        self.predelete_var   = BooleanVar(value=False)     # optionally run delete stage

        ttk.Checkbutton(
            self.root,
            text="Full update (include fixed libraries & mirror on device)",
            variable=self.full_update_var
        ).pack(padx=12, pady=(0,2), anchor="w")

        ttk.Checkbutton(
            self.root,
            text="Pre-delete staged paths on device (slower; extra flash wear)",
            variable=self.predelete_var
        ).pack(padx=12, pady=(0,6), anchor="w")

        self.log = ScrolledText(self.root, height=18, width=100)
        self.log.pack(fill='both', expand=True, padx=12, pady=(0,6))

        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(fill='both', padx=12, pady=(0,12))
        self.btn_start = ttk.Button(buttons_frame, text="Start", command=self.start)
        self.btn_cancel = ttk.Button(buttons_frame, text="Cancel", command=self.request_cancel, state="disabled")
        self.btn_start.pack(side="left")
        self.btn_cancel.pack(side="left", padx=8)

        # thread comms
        self.message_queue = queue.Queue()
        self.root.after(100, self.drain_message_queue)

    def log_line(self, s):
        self.log.insert(END, s + "\n")
        self.log.see(END)

    def set_status(self, s):
        self.status.set(s)

    def set_substatus(self, s):
        self.substatus.set(s)

    def tick(self, n=1):
        self.done_steps += n
        self.pbar["value"] = self.done_steps

    def request_cancel(self):
        self.cancel = True
        self.set_status("Cancelling after current step…")
        self.btn_cancel.config(state="disabled")

    def start(self):
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")
        full = self.full_update_var.get()
        predelete = self.predelete_var.get()
        t = threading.Thread(target=self.worker, args=(full, predelete), daemon=True)
        t.start()

    def drain_message_queue(self):
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                if kind == "log":
                    self.log_line(payload)
                elif kind == "status":
                    self.set_status(payload)
                elif kind == "substatus":
                    self.set_substatus(payload)
                elif kind == "tick":
                    self.tick(payload)
                elif kind == "done":
                    self.btn_cancel.config(state="disabled")
                    self.set_substatus("")
                    self.set_status(payload)
                    self.btn_start.config(state="normal")
                elif kind == "error":
                    self.log_line(payload)
        except queue.Empty:
            pass
        self.root.after(100, self.drain_message_queue)

    def worker(self, full_update, predelete):
        try:
            self.message_queue.put(("status", "Scanning ports…"))
            ports = tools.scan_for_ports()
            ports = tools.select_robot_ports(ports)

            # stage + (optional predelete) + upload
            steps_per_port = 2 + (1 if predelete else 0)
            steps_total = max(1, len(ports) * steps_per_port)
            self.total_steps = steps_total
            self.message_queue.put(("substatus", f"{len(ports)} robot(s) • full: {full_update} • predelete: {predelete}"))
            self.root.after(0, lambda: self.pbar.configure(maximum=steps_total))
            self.done_steps = 0
            self.root.after(0, lambda: self.pbar.configure(value=0))

            for port in ports:
                if self.cancel: break
                self.message_queue.put(("log", f"== {port} =="))

                # 1) Stage
                self.message_queue.put(("log", f"[{port}] staging ({'full' if full_update else 'app-only'})"))
                rshell.make_staging_copy(origin_folder, full=full_update)  # assume this bumps mtimes
                n_files = count_files(staging_folder)
                self.message_queue.put(("log", f"[{port}] {n_files} files staged"))
                self.message_queue.put(("tick", 1))

                # Send break to stop running app (so rshell can connect)
                self.message_queue.put(("log", f"[{port}] sending break"))
                try: tools.force_break(port)
                except Exception as e: self.message_queue.put(("log", f"[{port}] break error: {e}"))

                # 2) Optional pre-delete staged paths
                if predelete:
                    if self.cancel: break
                    note = " (redundant with mirror)" if full_update else ""
                    self.message_queue.put(("log", f"[{port}] removing existing files{note}"))
                    try:
                        rshell.remove_same(port, staging_folder, show=True)
                    except Exception as e:
                        self.message_queue.put(("log", f"[{port}] remove_same error: {e}"))
                    self.message_queue.put(("tick", 1))

                # 3) Upload
                if self.cancel: break
                self.message_queue.put(("status", f"[{port}] uploading"))
                self.message_queue.put(("log", f"[{port}] uploading"))
                try:
                    rshell.upload(port, staging_folder, mirror=full_update)  # mirror when full update
                except Exception as e:
                    self.message_queue.put(("log", f"[{port}] upload error: {e}"))
                self.message_queue.put(("tick", 1))
                self.message_queue.put(("log", f"[{port}] done"))

            msg = "Cancelled." if self.cancel else "All robots updated."
            self.message_queue.put(("done", msg))
        except Exception:
            self.message_queue.put(("error", traceback.format_exc()))

def main():
    ports = tools.select_robot_ports(tools.scan_for_ports())
    ui = ProgressUI(ports)
    ui.set_status("Ready to update")
    ui.root.mainloop()

if __name__ == "__main__":
    main()
