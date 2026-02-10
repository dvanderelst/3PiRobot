import socket
import struct
import msgpack
import numpy as np
import time
import select
from matplotlib import pyplot as plt

from Library import Settings
from Library import AcousticProcessing
from Library import Utils
from Library import FileOperations
from Library import Logging
from rich.console import Console
from rich.table import Table

def print_robot_timing(package, compare_ID=None):
    console = Console()
    acquire_id = package.get('acquire_id', None)
    requested = package.get('requested_fs_hz', None)
    effective = package.get('effective_fs_hz', None)
    sample_delay_us = package.get('sample_delay_us', None)
    emission_detected = package.get('emission_detected', None)
    total_duration_us = package.get('total_duration_us', None)
    robot_name = package['configuration'].robot_name

    id_matches = True
    if compare_ID is not None: id_matches = acquire_id == compare_ID

    table = Table(title=f"Robot Timing Report - {robot_name}")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Acquire ID", str(acquire_id))
    if not id_matches: table.add_row("[bold red]WARNING[/bold red]", "Acquire ID does not match!")
    table.add_row("Requested FS Hz", f"{requested:.2f}" if isinstance(requested, (int, float)) else str(requested))
    table.add_row("Effective FS Hz", f"{effective:.2f}" if isinstance(effective, (int, float)) else str(effective))
    table.add_row("Sample Delay (us)", str(sample_delay_us))
    table.add_row("Emission Detected", str(emission_detected))
    table.add_row("Total Duration (us)", str(total_duration_us))
    console.print(table)




class Client:
    def __init__(self, robot_number=0, ip=None):
        index = robot_number - 1
        configuration = Settings.get_client_config(index)
        self.configuration = configuration
        self.ip = self.configuration.ip
        if ip is not None: self.ip = ip
        self.sock = socket.socket()
        self.sock.settimeout(10)  # Increased from 5 to 10 seconds for better reliability
        self.sock.connect((self.ip, 1234))
        self.calibration = {}
        self.robot_number = robot_number
        self.load_calibration()

        self._samples_in_buffers = None

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

    def _send_dict(self, dct, require_ack=False):
        """
        Prefix-frame a msgpack dict and send it.
        
        Args:
            dct: Dictionary to send
            require_ack: If True, wait for acknowledgment before returning
            
        Returns:
            str: The acquire_id if require_ack=True and ACK received
            None: If require_ack=False or no ACK expected
        """
        packed = msgpack.packb(dct)
        prefix = struct.pack(">H", len(packed))
        self.sock.sendall(prefix + packed)
        
        # If acknowledgment is required, wait for it
        if require_ack:
            acquire_id = dct.get('acquire_id')
            if acquire_id:
                ack = self._wait_for_acknowledgment(acquire_id)
                if not ack:
                    raise TimeoutError(f"No acknowledgment received for command {acquire_id}")
                return acquire_id
        return None

    def _wait_for_message(self, predicate, timeout=10.0):
        end_time = time.time() + timeout
        while time.time() < end_time:
            if self.sock and select.select([self.sock], [], [], 0.1)[0]:
                msg = self._recv_msgpack()
                if msg and predicate(msg):
                    return msg
        return None

    def _send_command(self, action, params=None, wait_for_response=True, timeout=15.0, max_retries=1, retry_delay=0.2):
        """
        Send a command and optionally wait for its response.
        Retries the whole command if no response is received.
        """
        last_cmd_id = None
        for attempt in range(max_retries):
            cmd_id = Utils.make_code(n=8)
            last_cmd_id = cmd_id
            cmd = {'id': cmd_id, 'action': action}
            if params:
                cmd.update(params)
            self._send_dict(cmd)
            if not wait_for_response:
                return cmd_id
            resp = self._wait_for_message(lambda m: m.get('id') == cmd_id, timeout=timeout)
            if resp:
                return resp
            if attempt < max_retries - 1:
                backoff = retry_delay * (attempt + 1)
                self.print_message(f"No response for {action} (id={cmd_id}); retrying in {backoff:.1f}s", "WARNING")
                time.sleep(backoff)
        return None if wait_for_response else last_cmd_id

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
        self.calibration = calibration
        return calibration

    def set_kinematics(self, linear_speed=0, rotation_speed=0):
        """
        Drive with linear + rotational velocity (open-loop).
        """
        start = time.time()
        self._send_command(
            'kinematics',
            params={'linear_speed': linear_speed, 'rotation_speed': rotation_speed},
            wait_for_response=True,
            timeout=5.0
        )
        msg = f"Set_kinematics took {time.time() - start:.4f}s"
        self.print_message(msg, category='INFO')

    def stop_robot(self):
        """Convenience shortcut."""
        self.set_kinematics(0, 0)

    def change_robot_setting(self, parameter, value):
        """Change a setting on the robot."""
        start = time.time()
        parameter = str(parameter)
        self._send_command(
            'parameter',
            params={parameter: value},
            wait_for_response=True,
            timeout=5.0
        )
        msg = f"Changed settings in {time.time() - start:.4f}s"
        self.print_message(msg, category='INFO')

    def change_free_ping_period(self, period):
        """Change the free ping period on the robot."""
        self.change_robot_setting('free_ping_period', period)

    def step(self, distance=0, angle=0, linear_speed=0, rotation_speed=0, wait_for_completion=True, timeout=30.0, post_delay_s=0.05):
        start = time.time()
        angle = int(angle)
        params = {'distance': distance, 'angle': angle, 'linear_speed': linear_speed, 'rotation_speed': rotation_speed}
        if wait_for_completion:
            resp = self._send_command('step', params=params, wait_for_response=True, timeout=timeout, max_retries=1)
            if not resp or resp.get('status') != 'done':
                raise TimeoutError("No completion response for step")
            if post_delay_s and post_delay_s > 0:
                time.sleep(post_delay_s)
        else:
            self._send_command('step', params=params, wait_for_response=False)
        msg = f"Step (d={distance}, a={angle}) took {time.time() - start:.4f}s"
        self.print_message(msg, category='INFO')

    def acquire(self, action, wait_for_completion=True, timeout=15.0):
        # assure that action is either 'ping' or 'listen'
        error_message = f"Invalid action '{action}'. Action must be either 'ping' or 'listen'."
        if action not in ['ping', 'listen']: raise ValueError(error_message)
        start_time = time.time()
        sample_rate = self.configuration.sample_rate
        samples = self.configuration.samples

        params = {'sample_rate': sample_rate, 'samples': samples}
        if wait_for_completion:
            resp = self._send_command(action, params=params, wait_for_response=True, timeout=timeout, max_retries=3)
            if not resp:
                raise TimeoutError(f"No response received for {action}")
        else:
            resp = self._send_command(action, params=params, wait_for_response=False)
        self._samples_in_buffers = samples  # remember for later
        current_time = time.time()
        duration = (current_time - start_time) * 1000.0
        msg = f"Sent {action} command (sr={sample_rate}, samples={samples}) took {duration:.1f}ms"
        self.print_message(msg, category='INFO')
        return resp

    def _decode_sonar_response(self, sonar_package, plot=False):
        if sonar_package is None:
            return None
        if 'data' not in sonar_package or 'timing_info' not in sonar_package:
            return None

        samples = self._samples_in_buffers
        emitter_channel = self.configuration.emitter_channel
        left_channel = self.configuration.left_channel
        right_channel = self.configuration.right_channel

        start_time = time.time()
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
        duration = (current_time - start_time) * 1000.0
        msg = f"Reading buffers took {duration:.1f}ms"
        self.print_message(msg, category='INFO')

        effective_sample_rate = sonar_package['effective_fs_hz']
        raw_distance_axis = Utils.get_distance_axis(effective_sample_rate, samples)
        sonar_package['raw_distance_axis'] = raw_distance_axis
        # Add calibration is present, else empty dict
        sonar_package['calibration'] = {}
        if self.calibration: sonar_package['calibration'] = self.calibration
        # Add client configuration
        sonar_package['configuration'] = self.configuration
        if plot:
            plot_settings = {'y_max': 30000, 'y_min': 5000}
            AcousticProcessing.plot_sonar_data(raw_distance_axis, sonar_package, plot_settings)
            plt.show()
        return sonar_package

    def read_buffers(self, plot=False):
        self.print_message("read_buffers is not supported in the new protocol", "WARNING")
        return None

    def listen(self, plot=False):
        sonar_package = self.acquire(action='listen', wait_for_completion=True)
        sonar_package = self._decode_sonar_response(sonar_package, plot=plot)
        return sonar_package

    def read_and_process(self, do_ping=True, plot=False, close_after=False, selection_mode='first', wait_for_completion=True):
        calibration = self.calibration

        sonar_package = None
        if do_ping:
            sonar_package = self.acquire(action='ping', wait_for_completion=wait_for_completion)
            sonar_package = self._decode_sonar_response(sonar_package, plot=False)
        else:
            self.print_message("read_and_process requires do_ping=True in the new protocol", "WARNING")
        
        # Handle timeout case
        if sonar_package is None:
            self.print_message("No sonar data received - timeout occurred", "WARNING")
            return None

        # If no calibration, return the bare package so downstream can still inspect/plot raw
        if not calibration:
            self.print_message("No calibration loaded. Returning unprocessed data.", "WARNING")
            return sonar_package

        sonar_package = AcousticProcessing.locate_echo(sonar_package, selection_mode)
        sonar_package = AcousticProcessing.apply_correction(sonar_package)

        # Warnings if a correction wasn’t applied
        if not sonar_package.get('distance_correction_applied', False):
            self.print_message("No distance correction applied.", "WARNING")
        if not sonar_package.get('iid_correction_applied', False):
            self.print_message("No IID correction applied.", "WARNING")

        if plot:
            file_name = plot if isinstance(plot, str) else None
            AcousticProcessing.plot_sonar_package(sonar_package, file_name=file_name, close_after=close_after)

        return sonar_package
