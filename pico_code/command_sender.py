from machine import UART
import time

class UartCommandSender:
    def __init__(self, uart_id=0, baudrate=115200):
        self.uart = UART(uart_id, baudrate=baudrate)
        self.buffer = b""
        self.end_char='*'

    def send(self, msg):
        if not msg.endswith(self.end_char):
            msg += self.end_char
        self.uart.write(msg.encode())

    def receive(self, timeout_ms=1000):
        start = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            if self.uart.any():
                byte = self.uart.read(1)
                if byte:
                    self.buffer += byte
                    if self.end_char.encode() in self.buffer:
                        msg, _, self.buffer = self.buffer.partition(self.end_char.encode())
                        return msg.decode().strip()
        return None
