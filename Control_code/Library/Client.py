import socket
import struct
import msgpack
import numpy as np
import time
import pickle

from os import path

from Library import Process
from Library import Utils
from Library import ClientList
from Library import FileOperations
from Library import Logging

class Client:
    def __init__(self, robot_number=0, ip=None):
        index = robot_number - 1
        configuration = ClientList.get_config(index)
        self.configuration = configuration
        self.ip = self.configuration.ip
        if ip is not None: self.ip = ip
        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect((self.ip, 1234))
        self.calibration = self.load_calibration()

    def print_message(self, message, category="INFO"):
        robot_name = self.configuration.robot_name
        Logging.print_message(robot_name, message, category)

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

    def load_calibration(self):
        pass
        # todo: load and compare calibration
        # robot_name = self.configuration.robot_name
        # filename = FileOperations.get_calibration_file(robot_name)
        # file_exists = path.isfile(filename)
        # if not file_exists:
        #     self.print_message(f'Calibration file not found', category="WARNING")
        #     return None
        # # load calibration
        # with open(filename, 'rb') as f: calibration = pickle.load(f)
        # calibration_config = calibration['client_configuration']
        # # compare configurations
        # matches = Utils.compare_configurations(self.configuration, calibration_config)
        # self.print_message(f'Calibration file loaded (Matches: {matches})')
        # return calibration

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

    def change_free_ping_interval(self, interval):
        """Change the free ping interval on the robot."""
        self.change_robot_setting('free_ping_interval', interval)

    def step(self, distance=0, angle=0, linear_speed=0, rotation_speed=0):
        start = time.time()
        angle = int(angle)
        dictionary = {'action': 'step', 'distance': distance, 'angle': angle, 'linear_speed': linear_speed, 'rotation_speed': rotation_speed}
        self._send_dict(dictionary)
        self.print_message(f"step sent (d={distance}, a={angle}) in {time.time() - start:.4f}s")

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
        data = data.astype(np.float32)  # convert to float32 for processing
        timing_info = msg['timing_info']

        self.print_message(f"Ping took {time.time() - start:.4f}s")
        if plot: Utils.sonar_plot(data, sample_rate)
        distance_axis = Utils.get_distance_axis(sample_rate, samples)
        return data, distance_axis, timing_info

    def ping_process(self, cutoff_index = None, plot=False, close_after=False, selection_mode='first'):
        """Ping and run downstream processing."""
        data, distance_axis, timing_info = self.ping(plot=False)
        if cutoff_index is not None: data[cutoff_index:, :] = np.min(data)
        # data has channels in order: [emitter, left, right]
        if data is None: return None
        results = Process.process_sonar_data(data, self.baseline_function, self.configuration, selection_mode=selection_mode)
        results['cutoff_index'] = cutoff_index
        self.print_message('Data processed', category="INFO")

        file_name = None
        if isinstance(plot, str): file_name = plot
        if plot: Process.plot_processing(results, self.configuration, file_name=file_name, close_after=close_after)

        iid_correction = 0
        distance_coefficient = 1
        distance_intercept = 0
        if self.spatial_function is not None:
            iid_correction = self.spatial_function['iid_correction']
            distance_coefficient = self.spatial_function['distance_coefficient']
            distance_intercept = self.spatial_function['distance_intercept']

        raw_iid = results['raw_iid']
        raw_distance = results['raw_distance']
        corrected_distance = distance_intercept + distance_coefficient * raw_distance

        raw_distance_formatted = f"{raw_distance:.2f}"
        corrected_distance_formatted = f"{corrected_distance:.2f}"

        iid_formatted = f"{raw_iid:+.2f}"

        corrected_iid = raw_iid - iid_correction
        side_code = 'L' if corrected_iid < 0 else 'R'
        corrected_iid_formatted = f"{corrected_iid:+.2f}"

        raw_message = f"Rdist: {raw_distance_formatted} m, Riid: {iid_formatted}"
        corrected_message = f"Cdist: {corrected_distance_formatted} m, Ciid: {corrected_iid_formatted}, Side: {side_code}"
        message = f"{raw_message} | {corrected_message}"

        self.print_message(message, category="INFO")

        results['message'] = message
        results['side_code'] = side_code
        results['distance_coefficient'] = distance_coefficient
        results['distance_intercept'] = distance_intercept
        results['iid_correction'] = iid_correction
        results['corrected_distance'] = corrected_distance
        results['corrected_iid'] = corrected_iid

        return results



