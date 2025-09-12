from machine import UART, Pin
import settings
import time
import struct
import umsgpack as msgpack  # MicroPython-compatible

"""
Refactored ESP AT Wi‑Fi bridge for RP2040 (fixed 115200 baud)

Public API preserved:
  - setup_wifi(ssids=None) -> (bridge, ip, ssid) or (None, None, None)
  - class WifiServer with methods: setup(), scan_presets(), connect_wifi(), start_server(), get_ip(),
    read_messages(), send_data(), disable(), enable()

Internal reorganization:
  - Logger: leveled logging helper
  - UartIO: raw UART + helpers (CRLF writing, token reads, flush)
  - EspAT: ESP-specific control (resets, wait-ready, baud normalization, clean STA state)
  - WifiMgr: Wi‑Fi policy (scan, join, server)

Key behavior:
  - Fixed 115200 baud for runtime communication
  - Self-healing boot: hard reset → wait for ready or AT OK at 115200; if missing, probe common bauds once,
    set UART_DEF=115200, hard reset, and continue
  - Robust joins (stream read up to JOIN_TIMEOUT_MS, recognize WIFI CONNECTED/GOT IP/OK/FAIL/ERROR)
  - CWLAP scan with retries and timeouts
  - Safe, uniform command sending to avoid CRLF mistakes
"""

# -------------------- Tunables / constants --------------------
BOOT_BAUD = 74880
RUN_BAUD = 115200

# Timeouts (ms)
READY_TIMEOUT_MS = 8000
PING_PERIOD_MS = 250
SCAN_TIMEOUT_MS = 15000
SCAN_RETRIES = 2
SCAN_GAP_MS = 400
JOIN_TIMEOUT_MS = 20000
PROMPT_TIMEOUT_MS = 2000
READ_RESP_TIMEOUT_MS = 3000
FLUSH_DUR_MS = 60
HARD_RESET_OFF_MS = 200
HARD_RESET_BOOT_WAIT_S = 1.2

# Read tokens
TOK_OK = b"OK\r\n"
TOK_ERROR = b"ERROR\r\n"
TOK_SEND_OK = b"SEND OK"
TOK_READY_L = b"ready"      # lowercase sometimes
TOK_READY_U = b"Ready"

# Candidate bauds to probe when normalizing (ordered by likelihood)
PROBE_BAUDS = (RUN_BAUD, 921600, 460800, 230400, 115200, 9600, BOOT_BAUD)


# -------------------- Logging --------------------
class Logger:
    """Very small leveled logger.
    Levels: 0=QUIET, 1=INFO, 2=DEBUG, 3=WIRE
    """
    def __init__(self, verbose):
        # Map bool to level; accept int levels directly
        if isinstance(verbose, bool):
            self.level = 1 if verbose else 0
        else:
            try:
                self.level = int(verbose)
            except Exception:
                self.level = 1

    def info(self, *a):
        if self.level >= 1:
            print(*a)

    def debug(self, *a):
        if self.level >= 2:
            print(*a)

    def wire(self, *a):
        if self.level >= 3:
            print(*a)


# -------------------- UART I/O --------------------
class UartIO:
    def __init__(self, uart_id, tx_pin, rx_pin, logger):
        self.tx_pin = tx_pin
        self.rx_pin = rx_pin
        self.log = logger
        self.uart = UART(uart_id, baudrate=BOOT_BAUD, tx=tx_pin, rx=rx_pin)

    def open(self, baud):
        self.uart.init(baudrate=baud, tx=self.tx_pin, rx=self.rx_pin)
        self.flush()

    def flush(self, dur_ms=FLUSH_DUR_MS):
        t0 = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), t0) < dur_ms:
            if self.uart.any():
                _ = self.uart.read()

    def write(self, data):
        if isinstance(data, str):
            data = data.encode()
        self.uart.write(data)

    def write_cmd(self, cmd):
        """Write an AT command with CRLF appended (avoid literal \r\n bugs)."""
        if isinstance(cmd, bytes):
            data = cmd + b"\r\n"
        else:
            data = (cmd + "\r\n").encode()
        self.uart.write(data)
        self.log.wire(">", cmd)

    def any(self):
        return self.uart.any()

    def read(self):
        return self.uart.read()

    def read_until(self, tokens, timeout_ms=READ_RESP_TIMEOUT_MS):
        """Read bytes until one of the tokens appears or timeout.
        Returns bytes (may be empty)."""
        start = time.ticks_ms()
        out = b""
        while True:
            if self.any():
                out += self.read()
                for t in tokens:
                    if t in out:
                        return out
            if time.ticks_diff(time.ticks_ms(), start) > timeout_ms:
                return out


# -------------------- ESP AT control --------------------
class EspAT:
    def __init__(self, uartio: UartIO, en_pin: Pin, logger: Logger):
        self.io = uartio
        self.en_pin = en_pin
        self.log = logger

    # ---- Reset / ready ----
    def hard_reset(self, off_ms=HARD_RESET_OFF_MS, boot_wait_s=HARD_RESET_BOOT_WAIT_S):
        try:
            self.en_pin.value(0)
            time.sleep_ms(off_ms)
            self.en_pin.value(1)
        except Exception:
            pass
        time.sleep(boot_wait_s)

    def wait_ready(self, timeout_ms=READY_TIMEOUT_MS):
        start = time.ticks_ms()
        buf = b""
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            if self.io.any():
                buf += self.io.read()
                if TOK_READY_L in buf or TOK_READY_U in buf:
                    self.log.debug("[ESP] ready")
                    return True
            time.sleep_ms(20)
        self.log.debug("[ESP] ready timeout")
        return False

    def wait_ready_or_ping(self, timeout_ms=READY_TIMEOUT_MS, ping_period_ms=PING_PERIOD_MS):
        if self.wait_ready(timeout_ms=timeout_ms):
            return True
        # Fallback: ping for OK
        start = time.ticks_ms()
        buf = b""
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            self.io.write(b"AT\r\n")
            t0 = time.ticks_ms()
            while time.ticks_diff(time.ticks_ms(), t0) < ping_period_ms:
                if self.io.any():
                    buf += self.io.read()
                    if TOK_OK in buf:
                        self.log.debug("[ESP] AT OK (no 'ready' banner)")
                        return True
                time.sleep_ms(10)
        self.log.debug("[ESP] neither 'ready' nor AT OK within timeout")
        return False

    # ---- Sending helpers ----
    def send_cmd(self, cmd, wait_s=0.1):
        self.io.write_cmd(cmd)
        time.sleep(wait_s)
        return self.read_response()

    def read_response(self, timeout_ms=READ_RESP_TIMEOUT_MS):
        data = self.io.read_until((TOK_SEND_OK, TOK_OK, TOK_ERROR), timeout_ms=timeout_ms)
        try:
            return data.decode()
        except Exception:
            # lenient decode
            return ''.join(chr(b) for b in data if 32 <= b <= 126 or b in (10, 13))

    def ok(self, cmd, wait_s=0.2):
        return "OK" in self.send_cmd(cmd, wait_s)

    # ---- Baud normalization (one-time when needed) ----
    def resync_baud_and_normalize(self, use_restore=False):
        """Probe common bauds, then set persistent default to RUN_BAUD and hard reset.
        Returns True once a normalization+reset path was executed; False if probing failed.
        """
        for b in PROBE_BAUDS:
            self.io.open(b)
            self.io.write(b"AT\r\n")
            buf = self.io.read_until((TOK_OK,), timeout_ms=300)
            if TOK_OK in buf:
                self.log.info(f"[ESP] Found live baud: {b}")
                # minimize echo noise
                self.io.write_cmd("ATE0")
                _ = self.read_response(timeout_ms=500)
                if use_restore:
                    self.log.info("[ESP] RESTORE defaults (includes 115200)")
                    self.io.write_cmd("AT+RESTORE")
                    _ = self.read_response(timeout_ms=1000)
                else:
                    self.log.info("[ESP] Setting UART_DEF to 115200")
                    self.io.write_cmd("AT+UART_DEF=115200,8,1,0,0")
                    _ = self.read_response(timeout_ms=600)
                # Apply default via HARD reset
                self.hard_reset()
                # ROM banner is at 74880; flush it
                self.io.open(BOOT_BAUD)
                time.sleep(0.3)
                _ = self.io.read()
                # Now open at 115200 and verify
                self.io.open(RUN_BAUD)
                if self.wait_ready_or_ping(timeout_ms=4000):
                    self.log.info("[ESP] Now alive at 115200")
                else:
                    # Try a final AT
                    self.io.write(b"AT\r\n")
                    _ = self.read_response(timeout_ms=500)
                return True
        return False

    # ---- Clean STA state ----
    def ensure_clean_sta(self):
        self.ok('ATE0')
        self.ok('AT+CWMODE=1')
        self.ok('AT+CWDHCP=1,1')
        self.ok('AT+CWAUTOCONN=0')
        self.ok('AT+CWQAP')
        self.io.flush()


# -------------------- Wi‑Fi manager (scan/join/server) --------------------
class WifiMgr:
    def __init__(self, esp: EspAT, logger: Logger):
        self.esp = esp
        self.io = esp.io
        self.log = logger

    def scan_presets(self, ssids=None, timeout_ms=SCAN_TIMEOUT_MS, details=False, retries=SCAN_RETRIES, gap_ms=SCAN_GAP_MS):
        if ssids is None:
            ssids = list(settings.ssid_list.keys())
        self.esp.ensure_clean_sta()
        found_all = []
        for attempt in range(retries + 1):
            if self.log.level >= 1:
                print(f"[ESP] CWLAP attempt {attempt+1}/{retries+1}")
            self.io.write_cmd('AT+CWLAP')
            start = time.ticks_ms()
            buf = b""
            while True:
                if self.io.any():
                    buf += self.io.read()
                    if b"\r\nOK\r\n" in buf or b"\r\nERROR\r\n" in buf:
                        break
                if time.ticks_diff(time.ticks_ms(), start) > timeout_ms:
                    if self.log.level >= 1:
                        print("[ESP] CWLAP timeout (unfiltered)")
                    break
            try:
                txt = buf.decode()
            except Exception:
                txt = ''.join(chr(b) for b in buf if 32 <= b <= 126 or b in (10, 13))
            nets = cwlapparse(txt)
            if nets:
                found_all = nets
                break
            time.sleep_ms(gap_ms)
        if not found_all:
            return [] if not details else []
        wanted = set(ssids)
        filtered = [n for n in found_all if n.get('ssid') in wanted]
        filtered.sort(key=lambda d: d.get('rssi', -999), reverse=True)
        return filtered if details else [n['ssid'] for n in filtered]

    def connect_wifi(self, ssid, password, join_timeout_ms=JOIN_TIMEOUT_MS):
        self.esp.ensure_clean_sta()
        if self.log.level >= 1:
            print(f"[WiFi] Joining {ssid} ...")
        # Proper CRLF
        self.io.write(('AT+CWJAP="%s","%s"\r\n' % (ssid, password)).encode())
        start = time.ticks_ms()
        buf = b""
        got_ip = False
        while time.ticks_diff(time.ticks_ms(), start) < join_timeout_ms:
            if self.io.any():
                buf += self.io.read()
                if b"WIFI CONNECTED" in buf or b"GOT IP" in buf:
                    got_ip = True
                if b"\r\nOK\r\n" in buf:
                    return self.esp.send_cmd('AT+CIFSR', wait_s=0.3)
                if b"\r\nFAIL\r\n" in buf or b"\r\nERROR\r\n" in buf:
                    if self.log.level >= 1:
                        try:
                            print("[WiFi] Join failed:\n", buf.decode())
                        except Exception:
                            print("[WiFi] Join failed (binary len:", len(buf), ")")
                    return "ERROR"
            time.sleep_ms(20)
        if self.log.level >= 1:
            print("[WiFi] Join timeout")
        return "ERROR"

    def start_server(self, port=1234):
        self.esp.send_cmd('AT+CIPMUX=1')
        return self.esp.send_cmd('AT+CIPSERVER=1,%d' % port)

    def get_ip(self):
        resp = self.esp.send_cmd('AT+CIFSR')
        for ln in resp.splitlines():
            if ln.startswith('+CIFSR:STAIP'):
                return ln.split('"')[1]
        return None


# -------------------- Public class (shim over refactor) --------------------
class WifiServer:
    """
    Backward-compatible facade exposing the previous methods while using the
    refactored internals.
    """
    def __init__(self):
        self.tx_pin = Pin(settings.tx_pin)
        self.rx_pin = Pin(settings.rx_pin)
        self.en_pin = Pin(settings.en_pin, Pin.OUT)
        self.log = Logger(getattr(settings, 'verbose', 1))

        self.uartio = UartIO(1, self.tx_pin, self.rx_pin, self.log)
        self.esp = EspAT(self.uartio, self.en_pin, self.log)
        self.wifi = WifiMgr(self.esp, self.log)

        self._buf = b""  # for +IPD framing

    # ---------- lifecycle ----------
    def disable(self):
        self.log.info("[ESP] OFF")
        try:
            self.en_pin.value(0)
        except Exception:
            pass

    def enable(self):
        self.log.info("[ESP] ON")
        try:
            self.en_pin.value(1)
        except Exception:
            pass
        time.sleep(HARD_RESET_BOOT_WAIT_S)

        # Optional boot banner capture at 74880
        self.uartio.open(BOOT_BAUD)
        time.sleep(0.5)
        _ = self.uartio.read() if self.uartio.any() else None

        # Always talk at fixed run baud
        self.uartio.open(RUN_BAUD)
        self.log.info(f"[ESP] Using fixed baud: {RUN_BAUD}")

        # Hard reset first (most reliable), then verify at 115200
        self.esp.hard_reset()
        self.uartio.open(RUN_BAUD)
        if not self.esp.wait_ready_or_ping(timeout_ms=READY_TIMEOUT_MS):
            self.log.info("[ESP] Not alive at 115200 after hard reset — normalizing defaults")
            ok = self.esp.resync_baud_and_normalize(use_restore=False)
            if not ok:
                self.log.info("[ESP] DEF normalize failed; trying RESTORE")
                _ = self.esp.resync_baud_and_normalize(use_restore=True)

        # Clean init
        self.esp.ensure_clean_sta()
        ok = 'OK' in self.esp.send_cmd('AT', wait_s=0.3)
        self.log.info("[ESP] AT check:", "OK" if ok else "FAIL")

    def setup(self):
        """Clean start in STA mode, no auto-reconnect, no association."""
        self.disable()
        time.sleep(1)
        self.enable()
        time.sleep(0.5)

        # Soft reset then ready-or-ping for good measure
        self.esp.send_cmd('AT+RST', wait_s=0.2)
        self.esp.wait_ready_or_ping(timeout_ms=READY_TIMEOUT_MS)
        self.uartio.flush()

        self.esp.ensure_clean_sta()

    # ---------- public Wi‑Fi API (delegates) ----------
    def scan_presets(self, *a, **kw):
        return self.wifi.scan_presets(*a, **kw)

    def connect_wifi(self, *a, **kw):
        return self.wifi.connect_wifi(*a, **kw)

    def start_server(self, *a, **kw):
        return self.wifi.start_server(*a, **kw)

    def get_ip(self):
        return self.wifi.get_ip()

    # ---------- Framing / Messaging ----------
    def _wait_for_prompt(self, timeout_ms=PROMPT_TIMEOUT_MS):
        start = time.ticks_ms()
        buf = b""
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            if self.uartio.any():
                buf += self.uartio.read()
                if b'>' in buf:
                    return True
        return False

    def read_messages(self):
        """Parse +IPD frames, then unpack each payload using 2-byte msgpack length prefix."""
        msgs = []
        if self.uartio.any():
            self._buf += self.uartio.read()
        while b"+IPD," in self._buf:
            i = self._buf.find(b"+IPD,")
            chunk = self._buf[i:]
            if b":" not in chunk:
                break
            header, rest = chunk.split(b":", 1)
            try:
                length = int(header.split(b",")[-1])
            except Exception:
                self._buf = b""
                break
            if len(rest) < length:
                break
            payload = rest[:length]
            self._buf = rest[length:]
            if len(payload) < 2:
                continue
            plen = struct.unpack(">H", payload[:2])[0]
            if len(payload) < 2 + plen:
                continue
            body = payload[2:2+plen]
            try:
                msgs.append(msgpack.loads(body))
            except Exception as e:
                self.log.debug("[ESP] msgpack.loads error:", e)
        return msgs

    def send_data(self, obj, conn_id=0, max_chunk_size=1460):
        """Send a msgpack-encoded object in chunks with 2-byte length prefix per chunk."""
        try:
            packed = msgpack.dumps(obj)
        except Exception as e:
            self.log.debug("[ESP] Packing failed:", e)
            return False
        prefix = struct.pack(">H", len(packed))
        full = prefix + packed
        total_len, index, chunk_id, success = len(full), 0, 0, True
        while index < total_len:
            chunk = full[index:index+max_chunk_size]
            chunk_len = len(chunk)
            # Send AT+CIPSEND command for this chunk
            self.uartio.write(("AT+CIPSEND=%d,%d\r\n" % (conn_id, chunk_len)).encode())
            # Wait for '>' prompt (buffered)
            if not self._wait_for_prompt(timeout_ms=PROMPT_TIMEOUT_MS):
                self.log.debug(f"[ESP] Timed out waiting for '>' on chunk {chunk_id}")
                return False
            self.uartio.write(chunk)
            resp = self.esp.read_response()
            if TOK_SEND_OK.decode() not in resp:
                self.log.debug(f"[ESP] Chunk {chunk_id} failed to send.")
                self.log.debug("→", resp)
                success = False
                break
            index += chunk_len
            chunk_id += 1
        self.log.debug(f"[ESP] Data sent in {chunk_id} chunk(s), total {total_len} bytes")
        return success


# -------------------- Module-level convenience --------------------
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
            if getattr(settings, 'verbose', 1):
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


# -------------------- CWLAP parsing --------------------
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
            except Exception:
                continue
            mac = parts[3] if len(parts) >= 4 and parts[3] else None
            ch = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else None
            nets.append({'ssid': ssid, 'rssi': rssi, 'ecn': ecn, 'mac': mac, 'ch': ch})
    return nets
