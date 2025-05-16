import network
import socket
import time

class WifiServer:
    def __init__(self, ssid, password):
        self.ssid = ssid
        self.password = password
        self.port = 1234
        self.end_char = "*"
        self.addr = socket.getaddrinfo('0.0.0.0', self.port)[0][-1]
        self.sock = None
        self.conn = None
        self.client_addr = None
        self._buffer = b""

    def connect(self):
        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)
        if not wlan.isconnected():
            wlan.connect(self.ssid, self.password)
            for _ in range(10):
                if wlan.isconnected():
                    break
                time.sleep(1)
        if not wlan.isconnected():
            raise RuntimeError("Wi-Fi connection failed")
        print("Connected:", wlan.ifconfig())

    def start(self):
        self.sock = socket.socket()
        self.sock.bind(self.addr)
        self.sock.listen(1)
        print(f"Waiting for connection on port {self.port}...")
        self.conn, self.client_addr = self.sock.accept()
        print(f"Connected by {self.client_addr}")
        self._buffer = b""

    def receive(self):
        if not self.conn:
            return None
        while self.end_char.encode() not in self._buffer:
            data = self.conn.recv(1024)
            if not data:
                return None
            self._buffer += data
        msg, _, self._buffer = self._buffer.partition(self.end_char.encode())
        return msg.decode().strip()

    def send(self, msg):
        if self.conn:
            self.conn.send((msg + self.end_char).encode())

    def close(self):
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()
