from machine import UART, Pin
import settings
import time

def parse_command(command):
    split_char = settings.split_char
    parts = command.split(split_char)
    
    if parts[0] == 'motors':
        left = float(parts[1])
        right = float(parts[2])
        return parts[0], [left, right]
    
    if parts[0] == 'ping':
        rate = int(parts[1])
        samples = int(parts[2])
        return parts[0], [rate, samples]
    

class WifiServer:
    def __init__(self):
        self.baudrate = 115200
        self.tx_pin = Pin(16)
        self.rx_pin = Pin(17)
        self.en_pin = Pin(18, Pin.OUT)
        self.end_char = settings.end_char
        self.verbose = settings.verbose
        self.uart = UART(0, baudrate=self.baudrate, tx=self.tx_pin, rx=self.rx_pin)
        self.buffer = b""

    def enable(self):
        if self.verbose:
            print("[ESP] Powering on...")
        self.en_pin.value(1)
        time.sleep(1)

    def disable(self):
        if self.verbose:
            print("[ESP] Powering off...")
        self.en_pin.value(0)

    def send_cmd(self, cmd, wait=0.1):
        if self.verbose:
            print(f"[ESP →] {cmd}")
        self.uart.write(cmd + '\r\n')
        time.sleep(wait)
        response = self.read_response()
        if self.verbose:
            self.print_response(response)
        return response

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
                break  # timeout
        return out.decode()

    def print_response(self, response):
        lines = response.strip().splitlines()
        for line in lines:
            clean = line.strip()
            if clean:
                print(f"[ESP ←] {clean}")

    def connect_wifi(self, ssid, password):
        self.send_cmd('AT+RST', wait=1.5)
        self.send_cmd('ATE0')
        self.send_cmd('AT+CWMODE=1')
        self.send_cmd(f'AT+CWJAP="{ssid}","{password}"', wait=5)
        return self.send_cmd('AT+CIFSR')

    def start_server(self, port=1234):
        self.send_cmd('AT+CIPMUX=1')
        return self.send_cmd(f'AT+CIPSERVER=1,{port}')
        
    def get_ip(self):
        response = self.send_cmd("AT+CIFSR")
        lines = response.strip().splitlines()
        for line in lines:
            if line.startswith("+CIFSR:STAIP"):
                ip = line.split('"')[1]
                if self.verbose:
                    print(f"[ESP] IP Address: {ip}")
                return ip
        if self.verbose:
            print("[ESP] Failed to retrieve IP address.")
        return None
    
    def send_data(self, conn_id, data):
        if isinstance(data, str):
            data = data.encode()
        length = len(data)
        cmd = f"AT+CIPSEND={conn_id},{length}"
        if self.verbose:
            print(f"[ESP →] {cmd}")
        self.uart.write(cmd + '\r\n')
        # Wait for '>' prompt
        timeout = time.ticks_ms()
        while True:
            if self.uart.any():
                if b'>' in self.uart.read():
                    break
            if time.ticks_diff(time.ticks_ms(), timeout) > 2000:
                if self.verbose:
                    print("[ESP] Timed out waiting for '>'")
                return False
        # Send the actual data
        self.uart.write(data)
        if self.verbose:
            print(f"[ESP →] (data) {data.decode()}")
        # Optionally wait for SEND OK
        response = self.read_response()
        if self.verbose:
            self.print_response(response)
        return b"SEND OK" in response.encode()

    def wait_command(self):
        if self.verbose:
            print("[ESP] Waiting for incoming command...")
        buffer = b""
        payload = b""
        in_payload = False

        while True:
            if self.uart.any():
                buffer += self.uart.read()

                while b"+IPD," in buffer:
                    # Look for start of +IPD line
                    start = buffer.find(b"+IPD,")
                    buffer = buffer[start:]  # discard anything before

                    # Make sure we got the ':' separator
                    if b':' not in buffer:
                        break  # wait for more data

                    header, rest = buffer.split(b':', 1)
                    payload += rest
                    buffer = b""  # reset after extracting

                    in_payload = True  # now inside payload

            if in_payload and self.end_char.encode() in payload:
                command, _ = payload.split(self.end_char.encode(), 1)
                cmd_str = command.decode().strip()
                if self.verbose:
                    print(f"[ESP ←] Payload: {cmd_str}")
                return cmd_str

    
if __name__ == '__main__':
    wifi = WifiServer(verbose=True)

    print("[TEST] Enabling ESP8266...")
    wifi.enable()

    print("[TEST] Connecting to Wi-Fi...")
    wifi.connect_wifi("batnet", "lebowski")

    print("[TEST] Starting server...")
    wifi.start_server(1234)

    print("[TEST] Getting IP address...")
    ip = wifi.get_ip()
    print(f"[TEST] ESP8266 IP: {ip}")

    print("[TEST] Waiting for command from host (end with '#')...")
    cmd = wifi.wait_command()
    print(f"[TEST] Received command: '{cmd}'")

