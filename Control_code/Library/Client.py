import socket
import struct
import msgpack
import numpy as np
import time
import select
from matplotlib import pyplot as plt

import Library.Settings
from Library import Process
from Library import Utils
from Library import FileOperations
from Library import Logging


class Client:
    def __init__(self, robot_number=0, ip=None):
        index = robot_number - 1
        configuration = Library.Settings.get_client_config(index)
        self.configuration = configuration
        self.ip = self.configuration.ip
        if ip is not None: self.ip = ip
        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect((self.ip, 1234))
        self.calibration = self.load_calibration()

    def print_message(self, message, category):
        robot_name = self.configuration.robot_name
        Logging.print_message(robot_name, message, category)

    def close(self):
        self.sock.close()

    def _recv_exact_timed(self, n: int):
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        wait_s = 0.0
        read_s = 0.0
        timeout = self.sock.gettimeout()
        while got < n:
            t0 = time.perf_counter()
            r, _, _ = select.select([self.sock], [], [], timeout)
            wait_s += (time.perf_counter() - t0)
            if not r: return None, wait_s, read_s
            t1 = time.perf_counter()
            rcvd = self.sock.recv_into(mv[got:])
            read_s += (time.perf_counter() - t1)
            if rcvd == 0: return None, wait_s, read_s
            got += rcvd

        return mv, wait_s, read_s

    # def _recv_exact(self, n):
    #     """Receive exactly *n* bytes (or None on socket close)."""
    #     buf = b""
    #     while len(buf) < n:
    #         chunk = self.sock.recv(n - len(buf))
    #         if not chunk: return None
    #         buf += chunk
    #     return buf

    def _send_dict(self, dct):
        """Prefix-frame a msgpack dict and send it."""
        packed = msgpack.packb(dct)
        prefix = struct.pack(">H", len(packed))
        self.sock.sendall(prefix + packed)

    def _recv_msgpack(self):
        t0 = time.perf_counter()
        # 1) Header (2 bytes)
        pre, wait_h, read_h = self._recv_exact_timed(2)
        if not pre: return None
        t1 = time.perf_counter()
        length = struct.unpack(">H", pre)[0]
        # 2) Body (length bytes)
        body, wait_b, read_b = self._recv_exact_timed(length)
        if not body: return None
        t2 = time.perf_counter()
        # 3) Unpack
        msg = msgpack.unpackb(body, raw=False)
        t3 = time.perf_counter()

        # Diagnostics --> This code allows to see where time is spent
        first_byte_wait_ms = 1000.0 * wait_h  # time until ANY data arrived
        idle_wait_ms = 1000.0 * (wait_h + wait_b)  # total time waiting for readability
        read_ms = 1000.0 * (read_h + read_b)  # actual socket read time
        unpack_ms = 1000.0 * (t3 - t2)
        total_ms = 1000.0 * (t3 - t0)
        # Add timing info to the message
        msg['first_byte_wait_ms'] = first_byte_wait_ms
        msg['idle_wait_ms'] = idle_wait_ms
        msg['read_ms'] = read_ms
        msg['unpack_ms'] = unpack_ms
        msg['total_ms'] = total_ms
        return msg

    def load_calibration(self):
        """Load the calibration file for this robot."""
        robot_name = self.configuration.robot_name
        calibration = FileOperations.load_calibration(robot_name)
        return calibration

    def set_kinematics(self, linear_speed=0, rotation_speed=0):
        """
        Drive with linear + rotational velocity (open-loop).
        """
        start = time.time()
        dictionary = {'action': 'kinematics', 'linear_speed': linear_speed, 'rotation_speed': rotation_speed}
        self._send_dict(dictionary)
        msg = f"Set_kinematics took {time.time() - start:.4f}s"
        self.print_message(msg, category='INFO')

    def stop_robot(self):
        """Convenience shortcut."""
        self.set_kinematics(0, 0)

    def change_robot_setting(self, parameter, value):
        """Change a setting on the robot."""
        start = time.time()
        parameter = str(parameter)
        self._send_dict({'action': 'parameter', parameter: value})
        msg = f"Changed settings in {time.time() - start:.4f}s"
        self.print_message(msg, category='INFO')

    def change_free_ping_period(self, period):
        """Change the free ping period on the robot."""
        self.change_robot_setting('free_ping_period', period)

    def step(self, distance=0, angle=0, linear_speed=0, rotation_speed=0):
        start = time.time()
        angle = int(angle)
        dictionary = {'action': 'step', 'distance': distance, 'angle': angle, 'linear_speed': linear_speed, 'rotation_speed': rotation_speed}
        self._send_dict(dictionary)
        msg = f"Step (d={distance}, a={angle}) took {time.time() - start:.4f}s"
        self.print_message(msg, category='INFO')

    def acquire(self, action, plot=False, print_timing=False):
        # assure that action is either 'ping' or 'listen'
        error_message = f"Invalid action '{action}'. Action must be either 'ping' or 'listen'."
        if action not in ['ping', 'listen']: raise ValueError(error_message)
        start = time.time()
        sample_rate = self.configuration.sample_rate
        samples = self.configuration.samples

        emitter_channel = self.configuration.emitter_channel
        left_channel = self.configuration.left_channel
        right_channel = self.configuration.right_channel

        self._send_dict({'action': action, 'sample_rate': sample_rate, 'samples': samples})
        msg = self._recv_msgpack()
        self._send_dict({'action': 'acknowledge'})  # send ack back

        # Reshape and reorder data
        raw = msg['data']  # the single byte blob
        arr = np.frombuffer(raw, dtype='<u2')  # length = 3*samples
        a0, a1, a2 = np.split(arr, [samples, 2 * samples])
        data = np.column_stack([a0, a1, a2])  # shape: (samples, 3), same as before
        data = data[:, [emitter_channel, left_channel, right_channel]]  # reorder channels
        data = data.astype(np.float32)  # convert to float32 for processing
        robot_timing_info = msg['timing_info'] # This is a dict with timing info from the robot
        # Create messages and plot
        current_time = time.time()
        window = 1000 * (samples/sample_rate)
        duration = (current_time - start)*1000.0
        listen_msg = f"Listening for {window:.1f}ms took {duration:.1f}ms"
        ping_msg = f"Ping (Recording {window:.1f}ms) took {duration:.1f}ms"
        if action == 'listen': self.print_message(listen_msg, category='INFO')
        if action == 'ping': self.print_message(ping_msg, category='INFO')
        if plot: Utils.sonar_plot(data, sample_rate);plt.show()

        self.print_message(f"Robot timing info: {robot_timing_info}", category='DEBUG')
        self.print_message(f"First byte wait: {msg['first_byte_wait_ms']:.1f}ms", category='DEBUG')
        self.print_message(f"Idle wait: {msg['idle_wait_ms']:.1f}ms", category='DEBUG')
        self.print_message(f"Read time: {msg['read_ms']:.1f}ms", category='DEBUG')
        self.print_message(f"Unpack time: {msg['unpack_ms']:.1f}ms", category='DEBUG')
        self.print_message(f"Total time: {msg['total_ms']:.1f}ms", category='DEBUG')

        distance_axis = Utils.get_distance_axis(sample_rate, samples)
        return data, distance_axis, robot_timing_info

    def listen(self, plot=False):
        data, distance_axis, timing_info = self.acquire(action='listen', plot=plot)
        return data, distance_axis, timing_info

    def ping(self, plot=False):
        data, distance_axis, timing_info = self.acquire(action='ping', plot=plot)
        return data, distance_axis, timing_info

    def ping_process(self, plot=False, close_after=False, selection_mode='first'):
        results = {}
        calibration = self.calibration
        data, distance_axis, timing_info = self.ping(plot=False)
        results['data'] = data
        results['distance_axis'] = distance_axis
        results['timing_info'] = timing_info
        # In case no calibration is loaded, return unprocessed data
        if calibration == {}:
            message = "No calibration loaded. Returning unprocessed data."
            self.print_message(message, category='WARNING')
            return results

        # Detect the echo and get raw results
        file_name = None
        if isinstance(plot, str): file_name = plot
        raw_results = Process.locate_echo(self, data, calibration, selection_mode)
        if plot: Process.plot_locate_echo(raw_results, file_name, close_after,calibration)
        results.update(raw_results)

        # Try to correct the results based on the calibration

        # Distance calibration
        distance_present = calibration.get('distance_present', False)
        if distance_present:
            distance_coefficient = calibration['distance_coefficient']
            distance_intercept = calibration['distance_intercept']
            raw_distance = raw_results['raw_distance']
            corrected_distance = distance_intercept + distance_coefficient * raw_distance
            results['corrected_distance'] = corrected_distance
        else:
            message = "No distance calibration present. Not correcting distance."
            self.print_message(message, category='WARNING')

        # IID calibration
        iid_present = calibration.get('iid_present', False)
        if iid_present:
            zero_iids = calibration['zero_iids']
            mean_zero_iids = np.mean(zero_iids)
            raw_iid = raw_results['raw_iid']
            corrected_iid = raw_iid - mean_zero_iids
            side_code = 'L' if corrected_iid < 0 else 'R'
            results['corrected_iid'] = corrected_iid
            results['side_code'] = side_code
        else:
            message = "No IID calibration present. Not correcting IID"
            self.print_message(message, category='WARNING')
        messages = Process.create_messages(results)
        results.update(messages)
        return results

