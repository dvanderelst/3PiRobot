from machine import UART, Pin

class CommandSender:
    def __init__(self):
        uart_id = 0
        baudrate = 115200
        tx_pin = Pin(16)  # TX goes to robot RX
        rx_pin = Pin(17)  # RX comes from robot TX
        
        self.end_char = "*"
        self.uart = UART(uart_id, baudrate=baudrate, tx=tx_pin, rx=rx_pin)

    def send(self, msg):
        if not msg.endswith(self.end_char):
            msg += self.end_char
        self.uart.write(msg.encode())

    def receive(self):
        buffer = b""
        while True:
            if self.uart.any():
                byte = self.uart.read(1)
                buffer += byte
                if self.end_char.encode() in buffer:
                    msg, _, buffer = buffer.partition(self.end_char.encode())
                    return msg.decode().strip()
