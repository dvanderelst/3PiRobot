from machine import UART, Pin
import settings
import time
import struct
import umsgpack as msgpack  # MicroPython-compatible


def setup_wifi(ssids=None):
    """Return (bridge, ip, ssid) or (None, None, None) on failure."""
    print("[WiFi] Preparing module...")
    bridge = WifiServer()
    bridge.setup()

    # Robust scan (one wide scan + filter), with a quick retry
    for attempt in range(2):
        nets = bridge.scan_presets(ssids=ssids)
        print("[WiFi] Available networks:", nets)
        if nets:
            break
        if attempt == 0:
            if bridge.verbose:
                print("[WiFi] Rescanning after short pause...")
            time.sleep(0.5)

    if not nets:
        print("[WiFi] [FAIL] No known networks found")
        return None, None, None

    selected_ssid = nets[0]
    print("[WiFi] Connecting...")
    password = settings.ssid_list[selected_ssid]
    join_response = bridge.connect_wifi(selected_ssid, password)
    if "ERROR" in join_response:
        print(f"[WiFi] [FAIL] Could not join network '{selected_ssid}'")
        return None, None, None

    print(f"[WiFi] [OK] Connected to '{selected_ssid}'")
    print("        → IP info:")
    print(join_response)

    ip = bridge.get_ip()
    print('IP address:', ip)

    bridge.start_server(1234)
    return bridge, ip, selected_ssid


class WifiServer:
    """
    ESP AT Wi‑Fi bridge with robust baud negotiation and chunked send.
    - Captures boot log at 74,880.
    - Probes current AT baud, then attempts session-only switch to target.
    - Uses buffered wait for '>' prompt.
    - Sends MsgPack frames with a 2‑byte big‑endian length prefix.
    """

    def __init__(self):
        self.tx_pin = Pin(settings.tx_pin)
        self.rx_pin = Pin(settings.rx_pin)
        self.en_pin = Pin(settings.en_pin, Pin.OUT)

        self.boot_baud = 74880
        self.baudrate = 115200     # target baud when available
        # Create UART; exact baud will be set via _open_uart()
        self.uart = UART(1, baudrate=self.boot_baud, tx=self.tx_pin, rx=self.rx_pin)

        self.verbose = settings.verbose
        self._buf = b""

    # ---------- small helpers ----------
    def _flush_uart(self, dur_ms=60):
        t0 = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), t0) < dur_ms:
            if self.uart.any():
                _ = self.uart.read()

    def _open_uart(self, baud):
        """Re-init UART at a specific baud and flush boot noise."""
        self.uart.init(baudrate=baud, tx=self.tx_pin, rx=self.rx_pin)
        self._flush_uart()

    def _ok(self, cmd, wait=0.2):
        return "OK" in self.send_cmd(cmd, wait=wait)

    # ---------- baud helpers ----------
    def _probe_baud(self, candidates=(921600, 115200, 74880, 57600, 38400, 9600)):
        """Find a baud where 'AT' returns OK."""
        for b in candidates:
            self._open_uart(b)
            resp = self.send_cmd('AT', wait=0.15)
            if 'OK' in resp:
                if self.verbose:
                    print(f"[ESP] Found baud: {b}")
                return b
        return None

    def _set_cur_baud(self, target):
        """
        Try session-only change first (AT+UART_CUR).
        Fall back to persistent (AT+UART_DEF) if needed.
        """
        # Try CUR (session only)
        resp = self.send_cmd(f'AT+UART_CUR={target},8,1,0,0', wait=0.2)
        if 'OK' in resp:
            self._open_uart(target)
            ok = 'OK' in self.send_cmd('AT', wait=0.15)
            return ok

        # Fallback: DEF (persistent across resets)
        resp = self.send_cmd(f'AT+UART_DEF={target},8,1,0,0', wait=0.2)
        if 'OK' in resp:
            # Some firmwares need a reset after DEF
            self.send_cmd('AT+RST', wait=1.5)
            time.sleep(0.5)
            self._open_uart(target)
            ok = 'OK' in self.send_cmd('AT', wait=0.15)
            return ok

        return False

    # ---------- lifecycle ----------
    def disable(self):
        if self.verbose:
            print("[ESP] OFF")
        self.en_pin.value(0)

    def enable(self):
        if self.verbose:
            print("[ESP] ON")
        self.en_pin.value(1)
        time.sleep(1.2)

        # 1) Listen at boot baud (74,880) to capture boot log
        self._open_uart(self.boot_baud)
        time.sleep(1)
        if self.uart.any():
            log = self.uart.read()
            if self.verbose:
                try:
                    print("[ESP] Boot log:\n", log.decode())
                except:
                    print("[ESP] Boot log (raw):\n", log)

        # 2) Find the current AT baud and talk at that speed
        cur = self._probe_baud()
        if cur is None:
            if self.verbose:
                print("[ESP] Could not detect current baud; falling back to 115200")
            cur = 115200
            self._open_uart(cur)

        # 3) Switch to target (session-only if possible)
        if cur != self.baudrate:
            if self.verbose:
                print(f"[ESP] Switching baud {cur} -> {self.baudrate}")
            if not self._set_cur_baud(self.baudrate):
                if self.verbose:
                    print("[ESP] Baud switch failed; staying at", cur)
                # keep using 'cur' if switch failed
                self._open_uart(cur)

    def setup(self):
        """Clean start in STA mode, no auto-reconnect, no association."""
        self.disable()
        time.sleep(1)
        self.enable()
        time.sleep(1.5)

        self.send_cmd('AT+RST', wait=1.5)
        self._flush_uart()

        self._ok('ATE0')            # echo off
        self._ok('AT+CWMODE=1')     # STA
        self._ok('AT+CWAUTOCONN=0') # do not auto-reconnect
        self._ok('AT+CWQAP')        # ensure disconnected
        self._flush_uart()

    # ---------- AT helpers ----------
    def send_cmd(self, cmd, wait=0.1):
        if self.verbose > 1:
            print(f"> {cmd}")
        # ensure bytes on MicroPython
        if isinstance(cmd, str):
            cmd = cmd.encode()
        self.uart.write(cmd + b'\r\n')
        time.sleep(wait)
        return self._read_response()

    def _read_response(self, timeout_ms=3000):
        start = time.ticks_ms()
        out = b""
        while True:
            if self.uart.any():
                out += self.uart.read()
                # Break on common success tokens as well as ERROR
                if (b"SEND OK" in out) or (b"OK\r\n" in out) or (b"ERROR\r\n" in out):
                    break
            if time.ticks_diff(time.ticks_ms(), start) > timeout_ms:
                break
        try:
            return out.decode()
        except:
            # lenient decode for mixed binary/ASCII
            return ''.join(chr(b) for b in out if 32 <= b <= 126 or b in (10, 13))

    def _wait_for_prompt(self, timeout_ms=2000):
        """Wait for the '>' prompt from CIPSEND using a small buffer."""
        start = time.ticks_ms()
        buf = b""
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            if self.uart.any():
                buf += self.uart.read()
                if b'>' in buf:
                    return True
        return False

    # ---------- Scan / Connect ----------
    def scan_presets(self, ssids=None, timeout_ms=8000, details=False, retries=1, gap_ms=400):
        """
        Robust scan: single unfiltered CWLAP → parse → filter to presets.
        Returns list of SSID strings by default, or list of dicts if details=True.
        """
        if ssids is None:
            ssids = list(settings.ssid_list.keys())

        # be sure we are clean before scanning
        self._ok('AT+CWMODE=1')
        self._ok('AT+CWAUTOCONN=0')
        self._ok('AT+CWQAP')
        self._flush_uart()

        found_all = []
        for attempt in range(retries + 1):
            if self.verbose:
                print(f"[ESP] CWLAP attempt {attempt+1}/{retries+1}")
            self.uart.write(b'AT+CWLAP\r\n')
            start = time.ticks_ms()
            buf = b""
            while True:
                if self.uart.any():
                    buf += self.uart.read()
                    if b"\r\nOK\r\n" in buf or b"\r\nERROR\r\n" in buf:
                        break
                if time.ticks_diff(time.ticks_ms(), start) > timeout_ms:
                    if self.verbose:
                        print("[ESP] CWLAP timeout (unfiltered)")
                    break

            # lenient decode
            try:
                txt = buf.decode()
            except:
                txt = ''.join(chr(b) for b in buf if 32 <= b <= 126 or b in (10, 13))

            nets = cwlapparse(txt)
            if nets:
                found_all = nets
                break
            time.sleep_ms(gap_ms)

        if not found_all:
            return [] if not details else []

        # filter to presets
        wanted = set(ssids)
        filtered = [n for n in found_all if n.get('ssid') in wanted]
        filtered.sort(key=lambda d: d.get('rssi', -999), reverse=True)

        if details:
            return filtered
        else:
            return [n['ssid'] for n in filtered]

    def connect_wifi(self, ssid, password):
        # ensure clean state then join
        self._ok('AT+CWMODE=1')
        self._ok('AT+CWAUTOCONN=0')
        self._ok('AT+CWQAP')
        resp = self.send_cmd(f'AT+CWJAP="{ssid}","{password}"', wait=8)  # slightly longer wait
        if "OK" in resp or "WIFI CONNECTED" in resp:
            return self.send_cmd('AT+CIFSR')
        if self.verbose:
            print(f"[ESP] Wi-Fi failed:\n{resp}")
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

    # ---------- Framing / Messaging ----------
    def read_messages(self):
        """Parse +IPD frames, then unpack each payload using 2-byte msgpack length prefix."""
        msgs = []
        if self.uart.any():
            self._buf += self.uart.read()

        while b"+IPD," in self._buf:
            i = self._buf.find(b"+IPD,")
            chunk = self._buf[i:]
            if b":" not in chunk:
                break
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
            body = payload[2:2 + plen]
            try:
                msgs.append(msgpack.loads(body))
            except Exception as e:
                if self.verbose:
                    print("[ESP] msgpack.loads error:", e)
                # skip bad packet and continue
        return msgs

    def send_data(self, obj, conn_id=0, max_chunk_size=1460):
        """Send a msgpack-encoded object in chunks with 2-byte length prefix per chunk."""
        try:
            packed = msgpack.dumps(obj)
        except Exception as e:
            if self.verbose:
                print("[ESP] Packing failed:", e)
            return False

        prefix = struct.pack(">H", len(packed))
        full = prefix + packed

        total_len = len(full)
        index = 0
        chunk_id = 0
        success = True

        while index < total_len:
            chunk = full[index:index + max_chunk_size]
            chunk_len = len(chunk)

            # Send AT+CIPSEND command for this chunk
            self.uart.write(f"AT+CIPSEND={conn_id},{chunk_len}\r\n")

            # Wait for '>' prompt (buffered)
            if not self._wait_for_prompt(timeout_ms=2000):
                if self.verbose:
                    print(f"[ESP] Timed out waiting for '>' on chunk {chunk_id}")
                return False

            self.uart.write(chunk)
            resp = self._read_response()

            if "SEND OK" not in resp:
                if self.verbose:
                    print(f"[ESP] Chunk {chunk_id} failed to send.")
                    print("→", resp)
                success = False
                break

            index += chunk_len
            chunk_id += 1
            # No sleep needed; we wait for 'SEND OK'

        if self.verbose:
            print(f"[ESP] Data sent in {chunk_id} chunk(s), total {total_len} bytes")

        return success


def cwlapparse(text):
    """Parse AT+CWLAP output into a list of dicts."""
    nets = []
    for ln in text.splitlines():
        if not ln.startswith('+CWLAP:'):
            continue
        try:
            inner = ln[ln.index('(') + 1:ln.rindex(')')]
        except ValueError:
            continue
        # split respecting quotes
        parts, cur, inq = [], [], False
        for ch in inner:
            if ch == '"' and (not cur or cur[-1] != '\\'):
                inq = not inq
                continue
            if ch == ',' and not inq:
                parts.append(''.join(cur))
                cur = []
            else:
                cur.append(ch)
        parts.append(''.join(cur))

        if len(parts) >= 3:
            try:
                ecn = int(parts[0])
                ssid = parts[1]
                rssi = int(parts[2])
            except:
                continue
            mac = parts[3] if len(parts) >= 4 and parts[3] else None
            ch = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else None
            nets.append({'ssid': ssid, 'rssi': rssi, 'ecn': ecn, 'mac': mac, 'ch': ch})
    return nets
