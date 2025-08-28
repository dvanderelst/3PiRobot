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

class Client:
    def __init__(self, robot_number=0):
        index = robot_number - 1
        configuration = ClientList.get_config(index)
        self.configuration = configuration
        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect((self.configuration.ip, 1234))
        self.baseline_function = self.load_function('baseline')
        self.spatial_function = self.load_function('spatial')

    def print_message(self, message, category="INFO"):
        """Pretty‑print a log line with color and verbosity control."""
        robot_name = self.configuration.robot_name
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

    def load_function(self, function_name):
        filename = None
        robot_name = self.configuration.robot_name
        if function_name == 'spatial':
            filename = FileOperations.get_spatial_function_path(robot_name)
        if function_name == 'baseline':
            filename = FileOperations.get_baseline_function_path(robot_name)
        if filename is None: return None
        file_exists = path.isfile(filename)
        if not file_exists:
            self.print_message(f'Function ({function_name}) not found', category="WARNING")
            return None
        # load function
        with open(filename, 'rb') as f: loaded_function = pickle.load(f)
        loaded_function_config = loaded_function['client_configuration']
        # compare configurations
        matches = Utils.compare_configurations(self.configuration, loaded_function_config)
        self.print_message(f'Function ({function_name}) loaded (Matches: {matches})')
        return loaded_function

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

        # Prints all timing information - for debugging purposes
        #for k, v in timing_info.items(): print(f"{k}: {v}")

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



