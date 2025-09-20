from machine import Pin, ADC
import time
import array
import settings

DEBUG_TIMING = False  # set True if you want to include 'overruns' once in a while

def now():
    return time.ticks_us()

def time_since(previous_time):
    return time.ticks_diff(time.ticks_us(), previous_time)

class Sonar:
    def __init__(self, n_samples=0):
        self.n_samples = n_samples
        self.buf0 = None
        self.buf1 = None
        self.buf2 = None
        self.create_buffers()

        self.adc_emit  = ADC(settings.adc_emit)
        self.adc_recv1 = ADC(settings.adc_recv1)
        self.adc_recv2 = ADC(settings.adc_recv2)

        self.tgr_emit  = Pin(settings.trigger_emitter, Pin.OUT)
        self.tgr_recv1 = Pin(settings.trigger_recv1, Pin.OUT)
        self.tgr_recv2 = Pin(settings.trigger_recv2, Pin.OUT)

        self.tgr_emit.value(0); self.tgr_recv1.value(0); self.tgr_recv2.value(0)

        self.timeout_us = 100_000
        self.post_emit_settle_us = 20

        self.timing_info = {}

    def emit(self):
        self.tgr_recv1.value(0); self.tgr_recv2.value(0)
        self.tgr_emit.value(1); time.sleep_us(75); self.tgr_emit.value(0)

    def create_buffers(self):
        n = self.n_samples
        self.buf0 = array.array('H', (0 for _ in range(n)))
        self.buf1 = array.array('H', (0 for _ in range(n)))
        self.buf2 = array.array('H', (0 for _ in range(n)))

    def update_buffers(self, n_samples):
        if n_samples is not None and n_samples != self.n_samples:
            self.n_samples = n_samples
            self.create_buffers()

    def fill_buffers(self, sample_rate):
        """Deadline-scheduled sampling; store only effective fs (and overruns if DEBUG_TIMING)."""
        period_us_req = int(round(1_000_000 / int(sample_rate)))
        n = int(self.n_samples)

        next_due = time.ticks_us()
        t_first = None
        t_last = None
        overruns = 0

        for i in range(n):
            if time.ticks_diff(time.ticks_us(), next_due) > 0: overruns += 1
            while time.ticks_diff(time.ticks_us(), next_due) < 0: pass

            t_now = time.ticks_us()
            if t_first is None: t_first = t_now
            t_last = t_now

            self.buf0[i] = self.adc_emit.read_u16()
            self.buf1[i] = self.adc_recv1.read_u16()
            self.buf2[i] = self.adc_recv2.read_u16()

            next_due = time.ticks_add(next_due, period_us_req)

        # effective fs (average)
        intervals = max(1, n - 1)
        span_us = time.ticks_diff(t_last, t_first) if (t_first is not None and t_last is not None) else 0
        eff_period_us = (span_us / intervals) if span_us > 0 else 0
        eff_fs_hz = (1_000_000.0 / eff_period_us) if eff_period_us > 0 else 0.0
        self.timing_info['requested_fs_hz'] = sample_rate
        self.timing_info['effective_fs_hz'] = eff_fs_hz
        if DEBUG_TIMING: self.timing_info["overruns"] = overruns  # small int; optional

    def wait_for_emission(self):
        threshold = settings.pulse_threshold
        start = now()
        while True:
            if self.adc_emit.read_u16() > threshold: return True
            if time_since(start) > self.timeout_us: return False
            time.sleep_us(1)

    def acquire(self, mode, sample_rate=None, n_samples=None):
        mode = mode.lower()
        t0 = now()
        self.timing_info = {}

        if mode == 'pulse':
            self.emit()
            self.timing_info['mode'] = 'pulse'
            self.timing_info['total_duration_us'] = time_since(t0)

        self.update_buffers(n_samples)

        if mode == 'ping':
            self.timing_info['emission_delay_us'] = time_since(t0)
            self.emit()
            time.sleep_us(self.post_emit_settle_us)
            self.timing_info['emission_detected'] = self.wait_for_emission()

        if mode in ['ping', 'listen']:
            self.fill_buffers(sample_rate)


        self.timing_info['mode'] = mode
        self.timing_info['total_duration_us'] = time_since(t0)

