from machine import UART, Pin
import settings
import passwords
import time
import struct

# ──────────────────────────────────────────────────────
# Utilities

def array_to_bytes(arr):
    return struct.pack(f'{len(arr)}H', *arr)

def parse_command(command):
    split_char = settings.split_char
    parts = command.split(split_char)

    if parts[0] == 'motors':
        return parts[0], [float(parts[1]), float(parts[2])]
    if parts[0] == 'ping':
        return parts[0], [int(parts[1]), int(parts[2])]

    return [None, None]

def parse_commands(commands):
    return [parse_command(c) for c in commands]

# ──────────────────────────────────────────────────────
# WifiServer Class

class WifiServer:
    def __init__(self):
        self.tx_pin = Pin(settings.tx_pin)
        self.rx_pin = Pin(settings.rx_pin)
        self.en_pin = Pin(settings.en_pin, Pin.OUT)
        self.end_char = settings.end_char
        self.verbose = settings.verbose
        self.boot_baud = 74880
        self.baudrate = 115200
        self.uart = UART(1, baudrate=self.baudrate, tx=self.tx_pin, rx=self.rx_pin)
        self.buffer = b""

    def setup(self):
        self.disable()
        time.sleep(1)
        self.enable()
        time.sleep(1.5)
        self.send_cmd('AT+RST', wait=1.5)
        self.send_cmd('ATE0')
        self.send_cmd('AT+CWMODE=1')

    def disable(self):
        if self.verbose:
            print("[ESP] Powering off...")
        self.en_pin.value(0)

    def enable(self):
        if self.verbose:
            print("[ESP] Powering on...")
        self.en_pin.value(1)
        time.sleep(1.2)
        self.uart.init(baudrate=self.boot_baud, tx=self.tx_pin, rx=self.rx_pin)
        time.sleep(1)
        if self.uart.any():
            boot_log = self.uart.read()
            if self.verbose:
                try:
                    print("[ESP] Boot log:")
                    print(boot_log.decode())
                except:
                    print("[ESP] Boot log (raw):")
                    print(boot_log)
        self.uart.init(baudrate=self.baudrate, tx=self.tx_pin, rx=self.rx_pin)

    def connect_wifi(self, ssid, password):
        response = self.send_cmd(f'AT+CWJAP="{ssid}","{password}"', wait=6)
        if "OK" in response or "WIFI CONNECTED" in response:
            return self.send_cmd('AT+CIFSR')
        if self.verbose:
            print(f"[ESP] Failed to join {ssid}:")
            print(response)
        return "ERROR"

    def send_cmd(self, cmd, wait=0.1):
        if self.verbose:
            print(f"[ESP →] {cmd}")
        self.uart.write(cmd + '\r\n')
        time.sleep(wait)
        return self.read_response()

    def read_response(self):
        time.sleep(0.1)
        out = b""
        start_time = time.ticks_ms()
        while True:
            if self.uart.any():
                out += self.uart.read()
                if self.end_char.encode() in out:
                    break
            if time.ticks_diff(time.ticks_ms(), start_time) > 3000:
                break
        try:
            return out.decode()
        except UnicodeError:
            clean_bytes = bytes([b for b in out if 32 <= b <= 126 or b in (10, 13)])
            return clean_bytes.decode()

    def start_server(self, port=1234):
        self.send_cmd('AT+CIPMUX=1')
        return self.send_cmd(f'AT+CIPSERVER=1,{port}')

    def wait_commands(self):
        if self.verbose:
            print("[ESP] Waiting for incoming commands...")
        buffer = b""
        commands = []

        while True:
            if self.uart.any():
                buffer += self.uart.read()

                while b"+IPD," in buffer:
                    start = buffer.find(b"+IPD,")
                    buffer = buffer[start:]
                    if b":" not in buffer:
                        break
                    header, rest = buffer.split(b":", 1)
                    try:
                        length = int(header.split(b",")[-1])
                    except:
                        buffer = b""
                        break
                    if len(rest) < length:
                        break
                    payload = rest[:length]
                    buffer = rest[length:]
                    chunks = payload.split(self.end_char.encode())
                    for chunk in chunks[:-1]:
                        try:
                            cmd = chunk.decode().strip()
                            if cmd:
                                commands.append(cmd)
                        except:
                            pass
                    if not payload.endswith(self.end_char.encode()):
                        buffer = self.end_char.encode().join([chunks[-1]]) + buffer
            if commands:
                return commands

    def send_data(self, data, conn_id=0):
        if isinstance(data, str):
            data += self.end_char
            data = data.encode()
        length = len(data)
        self.uart.write(f"AT+CIPSEND={conn_id},{length}\r\n")
        timeout = time.ticks_ms()
        while True:
            if self.uart.any():
                if b'>' in self.uart.read():
                    break
            if time.ticks_diff(time.ticks_ms(), timeout) > 2000:
                return False
        self.uart.write(data)
        return b"SEND OK" in self.read_response().encode()

    def get_ip(self):
        response = self.send_cmd("AT+CIFSR")
        for line in response.strip().splitlines():
            if line.startswith("+CIFSR:STAIP"):
                return line.split('"')[1]
        return None

    def check_commands(self):
        if self.uart.any():
            return self.wait_commands()
        return []

# ──────────────────────────────────────────────────────
# Standalone Test

if __name__ == "__main__":
    print("[TEST] Starting Wi-Fi diagnostic test...")
    wifi = WifiServer()
    wifi.setup()

    ssid_list = settings.ssid_list
    use_ssid_index = 1
    ssid, password = ssid_list[use_ssid_index]

    print(f"[TEST] Attempting to join network: '{ssid}'")
    join_response = wifi.connect_wifi(ssid, password)

    if "ERROR" in join_response:
        print(f"[FAIL] Could not join network '{ssid}'")
    else:
        print(f"[OK] Successfully joined '{ssid}'")
        print("→ IP info:")
        print(join_response)

    print("[TEST] Starting server and waiting for incoming commands...")
    wifi.start_server(1234)
    commands = wifi.wait_commands()
    print("[TEST] Received commands:", commands)
