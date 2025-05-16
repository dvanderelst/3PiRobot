import socket

class Client:
    def __init__(self, host):
        self.host = host
        self.port = 1234
        self.end_char = "*"
        self.sock = socket.socket()
        self.sock.settimeout(5)
        self.sock.connect((self.host, self.port))
        self._buffer = b""

    def send(self, msg):
        if not msg.endswith(self.end_char):
            msg += self.end_char
        self.sock.send(msg.encode())

    def receive(self):
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
