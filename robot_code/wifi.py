from machine import UART, Pin
import settings
import time
import struct
import umsgpack as msgpack  # MicroPython‐compatible

class WifiServer:
    def __init__(self):
        self.tx_pin = Pin(settings.tx_pin)
        self.rx_pin = Pin(settings.rx_pin)
        self.en_pin = Pin(settings.en_pin, Pin.OUT)
        self.boot_baud = 74880
        self.baudrate = 115200
        self.uart = UART(1, baudrate=self.baudrate, tx=self.tx_pin, rx=self.rx_pin)
        self.verbose = settings.verbose
        self._buf = b""

    def setup(self):
        self.disable(); time.sleep(1)
        self.enable(); time.sleep(1.5)
        self.send_cmd('AT+RST', wait=1.5)
        self.send_cmd('ATE0')
        self.send_cmd('AT+CWMODE=1')

    def disable(self):
        if self.verbose: print("[ESP] OFF")
        self.en_pin.value(0)

    def enable(self):
        if self.verbose: print("[ESP] ON")
        self.en_pin.value(1)
        time.sleep(1.2)
        self.uart.init(baudrate=self.boot_baud, tx=self.tx_pin, rx=self.rx_pin)
        time.sleep(1)
        if self.uart.any():
            log = self.uart.read()
            if self.verbose:
                try: print("[ESP] Boot log:\n", log.decode())
                except: print("[ESP] Boot log (raw):\n", log)
        self.uart.init(baudrate=self.baudrate, tx=self.tx_pin, rx=self.rx_pin)

    def send_cmd(self, cmd, wait=0.1):
        if self.verbose > 1: print(f"> {cmd}")
        self.uart.write(cmd + '\r\n')
        time.sleep(wait)
        return self._read_response()

    def _read_response(self):
        start = time.ticks_ms()
        out = b""
        while True:
            if self.uart.any():
                out += self.uart.read()
                if b"OK\r\n" in out or b"ERROR\r\n" in out:
                    break
            if time.ticks_diff(time.ticks_ms(), start) > 3000:
                break
        try:
            return out.decode()
        except:
            return ''.join(chr(b) for b in out if 32 <= b <= 126 or b in (10,13))

    def connect_wifi(self, ssid, password):
        resp = self.send_cmd(f'AT+CWJAP="{ssid}","{password}"', wait=6)
        if "OK" in resp or "WIFI CONNECTED" in resp:
            return self.send_cmd('AT+CIFSR')
        if self.verbose: print(f"[ESP] Wi-fi failed:\n{resp}")
        return "ERROR"

    def start_server(self, port=1234):
        self.send_cmd('AT+CIPMUX=1')
        return self.send_cmd(f'AT+CIPSERVER=1,{port}')

    def get_ip(self):
        resp = self.send_cmd("AT+CIFSR")
        for ln in resp.splitlines():
            if ln.startswith('+CIFSR:STAIP'):
                return ln.split('"')[1]
        return None

    def read_messages(self):
        """Parse +IPD frames, then unpack each payload using 2-byte msgpack length prefix."""
        msgs = []
        if self.uart.any():
            self._buf += self.uart.read()

        while b"+IPD," in self._buf:
            i = self._buf.find(b"+IPD,")
            chunk = self._buf[i:]
            if b":" not in chunk: break
            header, rest = chunk.split(b":", 1)
            try:
                length = int(header.split(b",")[-1])
            except:
                self._buf = b"" 
                break
            if len(rest) < length:
                break  # wait for more data
            payload = rest[:length]
            self._buf = rest[length:]

            # payload = [2-byte prefix][msgpack data]
            if len(payload) < 2: 
                continue
            plen = struct.unpack(">H", payload[:2])[0]
            if len(payload) < 2 + plen:
                continue
            body = payload[2:2+plen]
            try:
                msgs.append(msgpack.loads(body))
            except Exception as e:
                if self.verbose: print("[ESP] msgpack.loads error:", e)
                # skip
        return msgs

    def send_data(self, obj, conn_id=0):
        """Prefix msgpack.dumps(obj) with 2-byte length, then AT+CIPSEND."""
        try:
            packed = msgpack.dumps(obj)
        except Exception as e:
            if self.verbose: print("[ESP] Packing failed:", e)
            return False

        prefix = struct.pack(">H", len(packed))
        full = prefix + packed
        self.uart.write(f"AT+CIPSEND={conn_id},{len(full)}\r\n")
        t0 = time.ticks_ms()
        while True:
            if self.uart.any() and b'>' in self.uart.read():
                break
            if time.ticks_diff(time.ticks_ms(), t0) > 2000:
                return False

        self.uart.write(full)
        resp = self._read_response()
        return "SEND OK" in resp
