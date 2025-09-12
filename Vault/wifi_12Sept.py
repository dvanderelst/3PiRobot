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

    # 0) If caller provided explicit SSIDs, try direct joins first
    if ssids:
        for s in ssids:
            if s in settings.ssid_list:
                print(f"[WiFi] Connecting to preset '{s}' (direct, no scan)...")
                join_response = bridge.connect_wifi(s, settings.ssid_list[s])
                if "ERROR" not in join_response:
                    print(f"[WiFi] [OK] Connected to '{s}'")
                    print("        → IP info:")
                    print(join_response)
                    ip = bridge.get_ip()
                    print('IP address:', ip)
                    bridge.start_server(1234)
                    return bridge, ip, s

    # 1) Robust scan (one wide scan + filter), with a quick retry
    nets = []
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
    ESP AT Wi-Fi bridge (fixed 115200 baud) with chunked send.
    - Captures boot log at 74,880 (optional).
    - Uses fixed 115200 local UART (no probing or switching).
    - Waits for 'ready' (or AT OK) after resets.
    - Sends MsgPack frames with a 2-byte big-endian length prefix.
    """

    def __init__(self):
        self.tx_pin = Pin(settings.tx_pin)
        self.rx_pin = Pin(settings.rx_pin)
        self.en_pin = Pin(settings.en_pin, Pin.OUT)

        self.boot_baud = 74880
        self.baudrate = 115200     # fixed, robust baud
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
        self.uart.init(baudrate=baud, tx=self.tx_pin, rx=self.rx_pin)
        self._flush_uart()

    def _ok(self, cmd, wait=0.2):
        return "OK" in self.send_cmd(cmd, wait=wait)

    def _wait_ready(self, timeout_ms=8000):
        """Wait for ESP 'ready' banner after boot/reset."""
        start = time.ticks_ms()
        buf = b""
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            if self.uart.any():
                buf += self.uart.read()
                if b"ready" in buf or b"Ready" in buf:
                    if self.verbose:
                        print("[ESP] ready")
                    return True
            time.sleep_ms(20)
        if self.verbose:
            print("[ESP] ready timeout")
        return False

    def _wait_ready_or_ping(self, timeout_ms=8000, ping_period_ms=250):
        """Wait for 'ready'; if not seen, periodically send 'AT' until we get 'OK'."""
        if self._wait_ready(timeout_ms=timeout_ms):
            return True
        # Fallback: ping for OK
        start = time.ticks_ms()
        buf = b""
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            self.uart.write(b'AT\r\n')
            t0 = time.ticks_ms()
            while time.ticks_diff(time.ticks_ms(), t0) < ping_period_ms:
                if self.uart.any():
                    buf += self.uart.read()
                    if b"OK\r\n" in buf:
                        if self.verbose:
                            print("[ESP] AT OK (no 'ready' banner)")
                        return True
                time.sleep_ms(10)
        if self.verbose:
            print("[ESP] neither 'ready' nor AT OK within timeout")
        return False

    def _resync_baud_and_normalize(self,
                                   candidates=(115200, 921600, 460800, 230400, 9600, 74880),
                                   probe_ms=300,
                                   use_restore=False):
        """
        Find the current live baud, then make 115200 the persistent default.
        If use_restore=True, issue AT+RESTORE instead of AT+UART_DEF.
        Always performs a HARD reset (EN toggle) before returning.
        """
        for b in candidates:
            self._open_uart(b)
            # probe with AT
            self.uart.write(b'AT\r\n')
            t0 = time.ticks_ms()
            buf = b""
            while time.ticks_diff(time.ticks_ms(), t0) < probe_ms:
                if self.uart.any():
                    buf += self.uart.read()
                    if b"OK\r\n" in buf:
                        if self.verbose:
                            print(f"[ESP] Found live baud: {b}")
                        # quiet the echo to reduce parsing noise
                        self.uart.write(b'ATE0\r\n');
                        time.sleep(0.05);
                        _ = self._read_response(400)

                        if use_restore:
                            if self.verbose: print("[ESP] RESTORE defaults (includes 115200)")
                            self.uart.write(b'AT+RESTORE\r\n')
                            # module will reboot; give it a moment
                            time.sleep(0.2)
                        else:
                            if self.verbose: print("[ESP] Setting UART_DEF to 115200")
                            self.uart.write(b'AT+UART_DEF=115200,8,1,0,0\r\n');
                            time.sleep(0.1)
                            _ = self._read_response(600)

                        # HARD reset so the default baud takes effect
                        self._hard_reset()
                        # After hard reset, the ROM boot banner is at 74880; flush it
                        self._open_uart(self.boot_baud)
                        time.sleep(0.3)
                        _ = self.uart.read() if self.uart.any() else None

                        # Now reopen at 115200 and confirm we can talk
                        self._open_uart(115200)
                        if self._wait_ready_or_ping(timeout_ms=4000):
                            if self.verbose:
                                print("[ESP] Now alive at 115200")
                            return True
                        # If we still can't see it, try one more AT ping
                        self.uart.write(b'AT\r\n');
                        _ = self._read_response(500)
                        return True  # we've done the normalization + hard reset
            time.sleep_ms(60)
        return False

    def _hard_reset(self, off_ms=200, boot_wait_s=1.2):
        """Toggle EN to force a true power-on reset."""
        try:
            self.en_pin.value(0)
            time.sleep_ms(off_ms)
            self.en_pin.value(1)
        except:
            # If EN is not wired, fall back silently
            pass
        time.sleep(boot_wait_s)

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

        # 1) (optional) read boot log at 74,880
        self._open_uart(self.boot_baud)
        time.sleep(0.5)
        _ = self.uart.read() if self.uart.any() else None

        # 2) switch to our preferred local baud
        self._open_uart(self.baudrate)  # 115200
        if self.verbose:
            print(f"[ESP] Using fixed baud: {self.baudrate}")

        # 3) reset and wait for ready or AT OK; if that fails, one-time resync
        # Prefer a HARD reset first so we aren't relying on the current baud
        self._hard_reset()
        # Try at our preferred rate
        self._open_uart(self.baudrate)  # 115200
        ok_ready = self._wait_ready_or_ping(timeout_ms=4000)

        if not ok_ready:
            if self.verbose:
                print("[ESP] Not alive at 115200 after hard reset — normalizing defaults")
            # Try DEF path first; if it still fails for you, flip use_restore=True
            ok = self._resync_baud_and_normalize(use_restore=False)
            if not ok and self.verbose:
                print("[ESP] DEF normalize failed; trying RESTORE")
                ok = self._resync_baud_and_normalize(use_restore=True)

        # 4) Clean init
        self._ok('ATE0')
        self._ok('AT+CWMODE=1')
        self._ok('AT+CWDHCP=1,1')
        self._ok('AT+CWAUTOCONN=0')
        self._ok('AT+CWQAP')

        ok = 'OK' in self.send_cmd('AT', wait=0.3)
        if self.verbose:
            print("[ESP] AT check:", "OK" if ok else "FAIL")

    def setup(self):
        """Clean start in STA mode, no auto-reconnect, no association."""
        self.disable()
        time.sleep(1)
        self.enable()
        time.sleep(0.5)

        self.send_cmd('AT+RST', wait=0.2)
        self._wait_ready_or_ping(timeout_ms=8000)
        self._flush_uart()

        self._ok('ATE0')
        self._ok('AT+CWMODE=1')
        self._ok('AT+CWDHCP=1,1')
        self._ok('AT+CWAUTOCONN=0')
        self._ok('AT+CWQAP')
        self._flush_uart()

    # ---------- AT helpers ----------
    def send_cmd(self, cmd, wait=0.1):
        if self.verbose > 1:
            print(f"> {cmd}")
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
    def scan_presets(self, ssids=None, timeout_ms=15000, details=False, retries=2, gap_ms=400):
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

    def connect_wifi(self, ssid, password, join_timeout_ms=20000):
        # ensure clean state then join
        self._ok('AT+CWMODE=1')
        self._ok('AT+CWAUTOCONN=0')
        self._ok('AT+CWQAP')

        if self.verbose:
            print(f"[WiFi] Joining {ssid} ...")

        # Send with real CRLF (not literal \r\n)
        self.uart.write(('AT+CWJAP="%s","%s"\r\n' % (ssid, password)).encode())

        start = time.ticks_ms()
        buf = b""
        got_ip = False

        while time.ticks_diff(time.ticks_ms(), start) < join_timeout_ms:
            if self.uart.any():
                buf += self.uart.read()
                if b"WIFI CONNECTED" in buf or b"GOT IP" in buf:
                    got_ip = True
                if b"\r\nOK\r\n" in buf:
                    return self.send_cmd('AT+CIFSR', wait=0.3)
                if b"\r\nFAIL\r\n" in buf or b"\r\nERROR\r\n" in buf:
                    if self.verbose:
                        try:
                            print("[WiFi] Join failed:\n", buf.decode())
                        except:
                            print("[WiFi] Join failed (binary len:", len(buf), ")")
                    return "ERROR"
            time.sleep_ms(20)

        if self.verbose:
            print("[WiFi] Join timeout")
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
