import socket
import struct
import msgpack
import numpy as np
import time
import Process
import Utils
import pickle
import matplotlib.pyplot as plt


class Client:
    def __init__(self, host, name=False):
        self.name     = name
        self.verbose  = True
        self.host     = host
        self.port     = 1234

        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect((self.host, self.port))

        self.baseline = self._load_baseline()

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
        if not pre:
            return None
        length = struct.unpack(">H", pre)[0]
        body = self._recv_exact(length)
        if not body:
            return None
        return msgpack.unpackb(body, raw=False)

    # ───────────────────────────── Motor / movement API ──────────────────────────

    def set_kinematics(self, linear_speed=0, rotation_speed=0):
        """
        Drive with linear + rotational velocity (open-loop).
        """
        start = time.time()
        self._send_dict(
            {
                'action': 'kinematics',
                'linear_speed': linear_speed,
                'rotation_speed': rotation_speed
            }
        )
        if self.verbose:
            print(f"set_kinematics took {time.time() - start:.4f}s")

    def set_motors(self, left, right):
        """
        Directly set raw left/right motor speeds.
        """
        start = time.time()
        self._send_dict({'action': 'motors', 'left': left, 'right': right})
        if self.verbose:
            print(f"set_motors took {time.time() - start:.4f}s")

    def stop(self):
        """Convenience shortcut."""
        self.set_motors(0, 0)

    def step(self, distance=0, angle=0, linear_speed=None, rotation_speed=None):
        start = time.time()
        self._send_dict(
            {
                'action': 'step',
                'distance': distance,
                'angle': angle,
                'linear_speed': linear_speed,
                'rotation_speed': rotation_speed,
            }
        )
        # server does not send a reply for step → nothing to read
        if self.verbose:
            print(f"step sent (d={distance}, a={angle}) in {time.time()-start:.4f}s")

    # ───────────────────────────── Sonar API ─────────────────────────────

    def _plot_raw_sonar_data(self, data, rate, samples):
        distance_axis = Utils.get_distance_axis(rate, samples)
        plt.plot(distance_axis, data)
        plt.title(self.name or self.host)
        plt.legend(['Emitter', 'Ch1', 'Ch2'])
        plt.xlabel("Distance (m)")
        plt.ylabel("Amplitude")
        plt.show()

    def ping(self, rate, samples, plot=False):
        """Fire sonar, return (data, distance_axis, timing_info)."""
        start = time.time()
        self._send_dict({'action': 'ping', 'rate': rate, 'samples': samples})

        msg = self._recv_msgpack()
        self._send_dict({'action': 'acknowledge'})  # send ack back

        data = np.array(msg['data'], dtype=np.uint16).reshape((3, samples)).T
        timing_info = msg['timing_info']

        if self.verbose:
            for k, v in timing_info.items():
                print(f"{k}: {v}")
            print(f"ping took {time.time() - start:.4f}s")

        if plot:
            self._plot_raw_sonar_data(data, rate, samples)

        distance_axis = Utils.get_distance_axis(rate, samples)
        return data, distance_axis, timing_info

    def ping_process(self, rate, samples, plot=False):
        """Ping and run downstream processing."""
        data, distance_axis, timing_info = self.ping(rate, samples, False)
        if data is None:
            return None
        results = Process.process(data, self.baseline, plot=plot)
        return results

    # ───────────────────────────── Baseline handling ─────────────────────────────

    def _load_baseline(self):
        filename = f'baselines/baseline_{self.host.replace(".", "_")}.pck'
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"[Baseline] File {filename} not found.")
            return None
        except Exception as e:
            print(f"[Baseline] Error reading {filename}: {e}")
            return None
