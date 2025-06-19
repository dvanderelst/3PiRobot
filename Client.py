import socket
import struct
import msgpack
import numpy as np
import time
import matplotlib.pyplot as plt

class Client:
    def __init__(self, host, name=False):
        self.verbose = True
        self.host = host
        self.port = 1234
        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect((self.host, self.port))
        self.name = name

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def send_dict(self, dct):
        packed = msgpack.packb(dct)
        prefix = struct.pack(">H", len(packed))
        self.sock.sendall(prefix + packed)

    def receive_msgpack(self):
        # Read 2-byte prefix
        pre = self._recv_exact(2)
        if not pre:
            return None
        length = struct.unpack(">H", pre)[0]
        body = self._recv_exact(length)
        if not body:
            return None
        return msgpack.unpackb(body, raw=False)

    def set_motors(self, left, right):
        start = time.time()
        self.send_dict({'action': 'motors', 'left': left, 'right': right})
        if self.verbose: print(f"set_motors took {time.time() - start:.4f}s")


    def plot_data(self, data, rate, samples):
        max_d = (343 / 2) * (samples / rate)
        dist = np.linspace(0, max_d, samples)
        plt.plot(dist, data)
        plt.title(self.name or self.host)
        plt.legend(['Emitter','Ch1','Ch2'])
        plt.xlabel("Distance (m)")
        plt.ylabel("Amplitude")
        plt.show()

    def ping_robust(self, rate, samples, plot=False):
        threshold_detection_delay = 0
        attempts = 0
        while threshold_detection_delay < 20000:
            data, timing_info = self.ping(rate, samples)
            threshold_detection_delay = timing_info.get('threshold detect (us)')
            attempts = attempts + 1
        timing_info['attempts'] = attempts
        if plot: self.plot_data(data, rate, samples)
        return data, timing_info

    def ping(self, rate, samples, plot=False):
        start = time.time()
        self.send_dict({'action': 'ping', 'rate': rate, 'samples': samples})
        msg = self.receive_msgpack()
        self.send_dict({'action': 'acknowledge'})  # ack back

        if msg is None or 'data' not in msg:
            print("No data received.")
            return None, None

        data = np.array(msg['data'], dtype=np.uint16).reshape((3, samples)).T
        #data -= data.min(axis=0)
        timing_info = msg['timing_info']

        if self.verbose:
            keys = timing_info.keys()
            for key in keys: print(key, timing_info[key])
            print(f"ping took {time.time() - start:.4f}s")

        if plot: self.plot_data(data, rate, samples)
        return data, timing_info

    def close(self):
        self.sock.close()
