import socket
import struct
import msgpack
import numpy as np
import time
import pickle

from Library import Process
from Library import Utils
from Library import ClientList



class Client:
    def __init__(self, index):
        configuration = ClientList.get_config(index)
        self.configuration = configuration
        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect((self.configuration.ip, 1234))
        self.baseline = self.load_baseline()

    # ───────────────────────────── function to handle messages ─────────────────────────────

    def print_message(self, message, category="INFO"):
        """Pretty‑print a log line with color and verbosity control."""
        robot_name = self.configuration.name
        verbose = self.configuration.verbose

        # --- configuration --------------------------------------------------
        LEVEL_THRESHOLD = {"ERROR": 0, "WARNING": 1, "INFO": 2}  # required verbose level
        COLOUR = {
            "ERROR":   "\033[91m",   # bright red
            "WARNING": "\033[93m",   # bright yellow
            "INFO":    "\033[94m",   # bright blue
        }
        NAME_STYLE   = "\033[1;96m"   # bold bright‑cyan
        RESET        = "\033[0m"
        # --------------------------------------------------------------------
        category = category.upper()
        # Skip if current verbosity is too low
        if verbose < LEVEL_THRESHOLD.get(category, 2): return
        cat_colour = COLOUR.get(category, "")
        print(f"{NAME_STYLE}[{robot_name}]{RESET} "
              f"{cat_colour}[{category}]{RESET} {message}")

    # ───────────────────────────── Low-level helpers ─────────────────────────────
    def close(self):
        self.sock.close()

    def _recv_exact(self, n):
        """Receive exactly *n* bytes (or None on socket close)."""
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _send_dict(self, dct):
        """Prefix-frame a msgpack dict and send it."""
        packed = msgpack.packb(dct)
        prefix = struct.pack(">H", len(packed))
        self.sock.sendall(prefix + packed)

    def _recv_msgpack(self):
        """Receive a single prefixed msgpack message (or None on error)."""
        pre = self._recv_exact(2)
        if not pre: return None
        length = struct.unpack(">H", pre)[0]
        body = self._recv_exact(length)
        if not body: return None
        return msgpack.unpackb(body, raw=False)

    # ───────────────────────────── Baseline loader  ─────────────────────────────
    def load_baseline(self):
        name = self.configuration.name
        filename = f'baselines/baseline_{name}.pck'
        try:
            with open(filename, 'rb') as f:
                loaded = pickle.load(f)
                self.print_message(f'File {filename} loaded')
                return loaded

        except FileNotFoundError:
            print(f"[Baseline] File {filename} not found.")
            return None
        except Exception as e:
            print(f"[Baseline] Error reading {filename}: {e}")
            return None

    def check_baseline_configuration(self):
        baseline_configuration = self.baseline['client_configuration']
        baseline_sample_rate = baseline_configuration.sample_rate
        baseline_samples = baseline_configuration.samples

        current_sample_rate = self.configuration.sample_rate
        current_samples = self.configuration.samples

        sample_rate_same = baseline_sample_rate == current_sample_rate
        samples_same = baseline_samples == current_samples
        matches = sample_rate_same and samples_same
        if not sample_rate_same:
            message = f"Sample rate mismatch: baseline {baseline_sample_rate}, current {current_sample_rate}"
            self.print_message(message, category="ERROR")
        if not samples_same:
            message = f"Samples mismatch: baseline {baseline_samples}, current {current_samples}"
            self.print_message(message, category="ERROR")
        if matches:
            message = "Baseline configuration matches current settings."
            self.print_message(message, category="INFO")
        return matches

    # ───────────────────────────── Motor / movement API ──────────────────────────

    def set_kinematics(self, linear_speed=0, rotation_speed=0):
        """
        Drive with linear + rotational velocity (open-loop).
        """
        start = time.time()
        dictionary = {'action': 'kinematics', 'linear_speed': linear_speed, 'rotation_speed': rotation_speed}
        self._send_dict(dictionary)
        self.print_message(f"Set_kinematics took {time.time() - start:.4f}s")

    def stop_robot(self):
        """Convenience shortcut."""
        self.set_kinematics(0, 0)

    def change_robot_setting(self, parameter, value):
        """Change a setting on the robot."""
        start = time.time()
        parameter = str(parameter)
        self._send_dict({'action': 'parameter', parameter: value})
        self.print_message(f"Changed settings in {time.time() - start:.4f}s")

    def step(self, distance=0, angle=0, linear_speed=0, rotation_speed=0):
        start = time.time()
        dictionary = {'action': 'step', 'distance': distance, 'angle': angle, 'linear_speed': linear_speed, 'rotation_speed': rotation_speed}
        self._send_dict(dictionary)
        self.print_message(f"step sent (d={distance}, a={angle}) in {time.time() - start:.4f}s")

    # ───────────────────────────── Sonar API ─────────────────────────────

    def ping(self, plot=False):
        """Fire sonar, return (data, distance_axis, timing_info)."""
        start = time.time()
        sample_rate = self.configuration.sample_rate
        samples = self.configuration.samples

        emitter_channel = self.configuration.emitter_channel
        left_channel = self.configuration.left_channel
        right_channel = self.configuration.right_channel

        self._send_dict({'action': 'ping', 'sample_rate': sample_rate, 'samples': samples})
        msg = self._recv_msgpack()
        self._send_dict({'action': 'acknowledge'})  # send ack back
        data = np.array(msg['data'], dtype=np.uint16).reshape((3, samples)).T
        data = data[:, [emitter_channel, left_channel, right_channel]]  # reorder channels
        timing_info = msg['timing_info']

        # Prints all timing information - for debugging purposes
        #for k, v in timing_info.items(): print(f"{k}: {v}")

        self.print_message(f"Ping took {time.time() - start:.4f}s")
        if plot: Utils.sonar_plot(data, sample_rate)
        distance_axis = Utils.get_distance_axis(sample_rate, samples)
        return data, distance_axis, timing_info

    def ping_process(self, plot=False):
        """Ping and run downstream processing."""
        data, distance_axis, timing_info = self.ping(plot=False)
        # data has channels in order: [emitter, left, right]
        self.check_baseline_configuration()
        if data is None: return None
        results = Process.process_sonar_data(data, self.baseline, self.configuration)
        self.print_message('Data processed', category="INFO")
        if plot: Process.plot_processing(results, self.configuration)

        iid = results['iid']
        distance = results['distance']

        iid_formatted = f"{iid:+.2f}"
        distance_formatted = f"{distance:.2f}"

        side = 'L' if iid < 0 else 'R'
        message = f"IID={iid_formatted} dB ({side}), Dist={distance_formatted} m"
        self.print_message(message, category="INFO")
        return results



