import socket
import struct
import msgpack
import numpy as np
import time

from matplotlib import pyplot as plt

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

    def change_free_ping_period(self, period):
        """Change the free ping period on the robot."""
        self.change_robot_setting('free_ping_period', period)

    def step(self, distance=0, angle=0, linear_speed=0, rotation_speed=0):
        start = time.time()
        angle = int(angle)
        dictionary = {'action': 'step', 'distance': distance, 'angle': angle, 'linear_speed': linear_speed, 'rotation_speed': rotation_speed}
        self._send_dict(dictionary)
        self.print_message(f"step sent (d={distance}, a={angle}) in {time.time() - start:.4f}s")

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
        msg = self._recv_msgpack()
        self._send_dict({'action': 'acknowledge'})  # send ack back
        data = np.array(msg['data'], dtype=np.uint16).reshape((3, samples)).T
        data = data[:, [emitter_channel, left_channel, right_channel]]  # reorder channels
        data = data.astype(np.float32)  # convert to float32 for processing
        timing_info = msg['timing_info']

        self.print_message(f"Ping took {time.time() - start:.4f}s")
        if plot:
            Utils.sonar_plot(data, sample_rate)
            plt.show()
        distance_axis = Utils.get_distance_axis(sample_rate, samples)
        return data, distance_axis, timing_info

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

