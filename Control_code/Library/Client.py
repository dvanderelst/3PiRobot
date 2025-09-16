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

    def acquire(self, action, plot=False):
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
        sonar_package = self._recv_msgpack()
        self._send_dict({'action': 'acknowledge'})  # send ack back
        # Flatten timing info into the main dict
        timing_info = sonar_package.pop('timing_info')
        sonar_package.update(timing_info)

        # Reshape and reorder data
        raw = sonar_package.pop('data')  # the single byte blob
        arr = np.frombuffer(raw, dtype='<u2')  # length = 3*samples
        a0, a1, a2 = np.split(arr, [samples, 2 * samples])
        sonar_data = np.column_stack([a0, a1, a2])  # shape: (samples, 3), same as before
        sonar_data = sonar_data[:, [emitter_channel, left_channel, right_channel]]  # reorder channels
        sonar_data = sonar_data.astype(np.float32)  # convert to float32 for processing
        sonar_package['sonar_data'] = sonar_data
        # Create messages and plot
        current_time = time.time()
        window = 1000 * (samples/sample_rate)
        duration = (current_time - start)*1000.0
        listen_msg = f"Listening for {window:.1f}ms took {duration:.1f}ms"
        ping_msg = f"Ping (Recording {window:.1f}ms) took {duration:.1f}ms"
        if action == 'listen': self.print_message(listen_msg, category='INFO')
        if action == 'ping': self.print_message(ping_msg, category='INFO')

        effective_sample_rate = sonar_package['effective_fs_hz']
        raw_distance_axis = Utils.get_distance_axis(effective_sample_rate, samples)
        sonar_package['raw_distance_axis'] = raw_distance_axis

        if plot: Utils.sonar_plot(sonar_package);plt.show()
        return sonar_package

    def listen(self, plot=False):
        sonar_package = self.acquire(action='listen', plot=plot)
        return sonar_package

    def ping(self, plot=False):
        sonar_package = self.acquire(action='ping', plot=plot)
        return sonar_package

    def ping_process(self, plot=False, close_after=False, selection_mode='first'):
        """
        1) Acquire a sonar_package (data + raw_distance_axis + timing).
        2) Run echo detection (locate_echo) using the loaded calibration.
        3) Apply distance/IID correction in one place (apply_correction).
        4) Warn if corrections weren’t applied.
        5) Optionally plot, then return the corrected result dict.
        """
        calibration = self.calibration

        # 1) Acquire (single dict: sonar_data, raw_distance_axis, timing, etc.)
        sonar_package = self.ping(plot=False)

        # If no calibration, return the bare package so downstream can still inspect/plot raw
        if not calibration:
            self.print_message("No calibration loaded. Returning unprocessed data.", "WARNING")
            return sonar_package

        # 2) Detect echo on this capture
        sonar_package = Process.locate_echo(self, sonar_package, calibration, selection_mode)

        # Ensure the sonar_package rides along (apply_correction expects it for axes)
        # This is already done inside locate_echo
        # if 'sonar_package' not in raw_results: raw_results['sonar_package'] = sonar_package

        # 3) Apply distance/IID correction (adds corrected_* fields; preserves raw)
        sonar_package = Process.apply_correction(sonar_package, calibration)

        # 4) Warnings if a correction wasn’t applied
        if not sonar_package.get('distance_correction_applied', False):
            self.print_message("No distance correction applied.", "WARNING")
        if not sonar_package.get('iid_correction_applied', False):
            self.print_message("No IID correction applied.", "WARNING")

        # Human-readable strings (will include corrected_* when present)
        # corrected.update(Process.create_messages(corrected))

        # 5) Optional plot
        if plot:
            file_name = plot if isinstance(plot, str) else None
            Process.plot_locate_echo(sonar_package, file_name=file_name, close_after=close_after, calibration=calibration)

        return sonar_package



