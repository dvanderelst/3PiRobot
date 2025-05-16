import network
import socket
import time

class WifiServer:
    def __init__(self, ssid, password, port=1234):
        self.ssid = ssid
        self.password = password
        self.port = port
        self.addr = socket.getaddrinfo('0.0.0.0', port)[0][-1]
        self.sock = None
        self.conn = None
        self.client_addr = None

    def connect(self):
        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)
        if not wlan.isconnected():
            print(f"Connecting to {self.ssid}...")
            wlan.connect(self.ssid, self.password)
            for _ in range(10):
                if wlan.isconnected():
                    break
                time.sleep(1)
        if wlan.isconnected():
            print("Connected:", wlan.ifconfig())
        else:
            raise RuntimeError("Wi-Fi connection failed")

    def start(self):
        self.sock = socket.socket()
        self.sock.bind(self.addr)
        self.sock.listen(1)
        print(f"Listening on port {self.port}...")
        self.conn, self.client_addr = self.sock.accept()
        print(f"Connected by {self.client_addr}")

    def read(self):
        if self.conn:
            data = self.conn.recv(1024)
            if data:
                return data.decode().strip()
        return None

    def send(self, msg):
        if self.conn:
            self.conn.send((msg + "\n").encode())

    def close(self):
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()
