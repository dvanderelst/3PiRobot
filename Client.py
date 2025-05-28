import socket
import struct
import numpy
import time

class Client:
    def __init__(self, host):
        self.verbose = True
        self.host = host
        self.port = 1234
        self.end_char = "*"
        self.split_char = ","
        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect((self.host, self.port))
        self._buffer = b""

    def set_motors(self, left, right):
        start = time.time()
        self.send_list(['motors', left, right])
        end = time.time()
        if self.verbose: print(f"set_motors took {end - start:.4f} seconds")

    def ping(self, rate, samples):
        start = time.time()
        self.send_list(['ping', rate, samples])
        number_of_bytes = samples * 2 * 2
        buffer =  self.receive(number_of_bytes)
        # Must be sent to acknowledge the data reception to avoid next command to be read as the response
        self.send_list(['received data'])
        data = struct.unpack(f"{samples * 2}H", buffer)
        data = numpy.array(data)
        data = data.reshape((2, samples))
        data = data.transpose()
        data[:, 0] = data[:, 0]  - numpy.min(data[:, 0])
        data[:, 1] = data[:, 1]  - numpy.min(data[:, 1])
        end = time.time()
        if self.verbose: print(f"ping took {end - start:.4f} seconds")
        max_distance = (343 / 2) * (samples / rate)
        distances = numpy.linspace(0, max_distance, samples)
        return data, distances

    def send_list(self, lst):
        msg = ''
        for item in lst: msg += f"{item}{self.split_char}"
        msg = msg.rstrip(self.split_char)
        self.send(msg)

    def send(self, msg):
        if not msg.endswith(self.end_char):
            msg += self.end_char
        self.sock.send(msg.encode())

    def receive(self, num_bytes=None):
        if num_bytes is not None:
            buffer = b""
            while len(buffer) < num_bytes:
                try:
                    chunk = self.sock.recv(num_bytes - len(buffer))
                    if not chunk:
                        return None
                    buffer += chunk
                except socket.timeout:
                    return None
            return buffer  # binary data

        # Otherwise, read until end_char (text mode)
        while self.end_char.encode() not in self._buffer:
            try:
                data = self.sock.recv(1024)
                if not data:
                    return None
                self._buffer += data
            except socket.timeout:
                return None
        msg, _, self._buffer = self._buffer.partition(self.end_char.encode())
        return msg.decode().strip()

    def query(self, msg):
        self.send(msg)
        return self.receive()

    def close(self):
        self.sock.close()
