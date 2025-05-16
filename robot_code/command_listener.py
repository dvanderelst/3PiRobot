from machine import UART, Pin

class CommandListener:
    def __init__(self):
        uart_id = 1
        baudrate = 115200
        tx_pin = Pin(4)
        rx_pin = Pin(5)
        
        self.end_char = "*"
        self.buffer = b""
        self.uart = UART(uart_id,baudrate=baudrate,tx=tx_pin,rx=rx_pin)


    def read(self):
        while self.uart.any():
            byte = self.uart.read(1)
            if byte:
                self.buffer += byte
                if self.end_char.encode() in self.buffer:
                    cmd, _, self.buffer = self.buffer.partition(self.end_char.encode())
                    return cmd.decode().strip()
        return None
    
    def wait_for_command(self):
        while True:
            byte = self.uart.read(1)
            if byte:
                self.buffer += byte
                if self.end_char.encode() in self.buffer:
                    cmd, _, self.buffer = self.buffer.partition(self.end_char.encode())
                    return cmd.decode().strip()

    def send(self, msg):
        if not msg.endswith(self.end_char):
            msg += self.end_char
        self.uart.write(msg.encode())
