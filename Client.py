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
        self.name = name
        self.verbose = True
        self.host = host
        self.port = 1234
        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect((self.host, self.port))
        self.baseline = self.load_baseline()

    def close(self):
        self.sock.close()

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk: return None
            buf += chunk
        return buf

    def send_dict(self, dct):
        packed = msgpack.packb(dct)
        prefix = struct.pack(">H", len(packed))
        self.sock.sendall(prefix + packed)

    def receive_msgpack(self):
        # Read 2-byte prefix
        pre = self._recv_exact(2)
        if not pre: return None
        length = struct.unpack(">H", pre)[0]
        body = self._recv_exact(length)
        if not body: return None
        return msgpack.unpackb(body, raw=False)

    def set_kinematics(self, linear_speed=0, rotational_speed=0):
        start = time.time()
        self.send_dict({'action': 'kinematics', 'linear_speed': linear_speed, 'rotational_speed': rotational_speed})
        if self.verbose: print(f"set_kinematics took {time.time() - start:.4f}s")

    def set_motors(self, left, right):
        start = time.time()
        self.send_dict({'action': 'motors', 'left': left, 'right': right})
        if self.verbose: print(f"set_motors took {time.time() - start:.4f}s")

    def stop(self):
        self.set_motors(0, 0)

    def plot_raw_sonar_data(self, data, rate, samples):
        distance_axis = Utils.get_distance_axis(rate, samples)
        plt.plot(distance_axis, data)
        plt.title(self.name or self.host)
        plt.legend(['Emitter', 'Ch1', 'Ch2'])
        plt.xlabel("Distance (m)")
        plt.ylabel("Amplitude")
        plt.show()

    def ping(self, rate, samples, plot=False):
        start = time.time()
        self.send_dict({'action': 'ping', 'rate': rate, 'samples': samples})
        msg = self.receive_msgpack()
        self.send_dict({'action': 'acknowledge'})  # ack back
        data = np.array(msg['data'], dtype=np.uint16).reshape((3, samples)).T
        timing_info = msg['timing_info']

        if self.verbose:
            keys = timing_info.keys()
            for key in keys: print(key, timing_info[key])
            print(f"ping took {time.time() - start:.4f}s")

        distance_axis = Utils.get_distance_axis(rate, samples)
        if plot: self.plot_raw_sonar_data(data, rate, samples)
        return data, distance_axis, timing_info

    def ping_process(self, rate, samples, plot=False):
        data, distance_axis, timing_info = self.ping(rate, samples, False)
        if data is None: return None
        results = Process.process(data, self.baseline, plot=plot)
        # if self.verbose:
        #     print('Onset distance:', results['distance'])
        #     print('Log integrals:', results['log_integrals'])
        #     print('Inter-channel difference (IID):', results['iid'])
        # return results

    def load_baseline(self):
        baseline_filename = f'baselines/baseline_{self.host.replace(".", "_")}.pck'
        try:
            with open(baseline_filename, 'rb') as f:
                baseline_data = pickle.load(f)
            return baseline_data
        except FileNotFoundError:
            print(f"Baseline file {baseline_filename} not found.")
            return None
        except Exception as e:
            print(f"Error reading baseline file: {e}")
            return None

