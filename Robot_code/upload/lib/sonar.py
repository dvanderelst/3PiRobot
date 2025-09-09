from machine import Pin, ADC
import time
import array
import settings

# ─────────────────────────────────────────────
# unified acquisition
# ─────────────────────────────────────────────

def emit_pulse(pin):
    pin.value(0)
    time.sleep_us(10)
    pin.value(1)
    time.sleep_us(50)
    pin.value(0)

def acquire(mode, sample_rate=10000, n_samples=100):
    """
    mode: 'ping' | 'pulse' | 'listen'
    sample_rate (Hz): used for 'ping' and 'listen'
    n_samples: number of samples for 'ping' and 'listen'

    Returns (buf0, buf1, buf2, timing_info)
      - buf0 : ADC settings.adc_recv1
      - buf1 : ADC settings.adc_recv2
      - buf2 : ADC settings.adc_emitter  (emitter monitor channel)
      - timing_info: dict with timestamps/diagnostics
      - For 'pulse': buf0, buf1, buf2 are all None
    """
    timeout_us = 100_000
    post_emit_settle_us = 20
    threshold = settings.pulse_threshold

    mode = str(mode).lower()

    # Emitter pin
    trigger_emitter = Pin(settings.trigger_emitter, Pin.OUT)
    # Keep MaxBotix receivers in receive-only mode
    Pin(settings.trigger_recv1, Pin.OUT).value(0)
    Pin(settings.trigger_recv2, Pin.OUT).value(0)

    # ADCs
    adc_emit  = ADC(settings.adc_emit)
    adc_recv1 = ADC(settings.adc_recv1)
    adc_recv2 = ADC(settings.adc_recv2)

    timing_info = {}
    t_start = time.ticks_us()

    # ── Mode: pulse ──
    if mode == 'pulse':
        emit_pulse(trigger_emitter)
        timing_info['mode'] = 'pulse'
        timing_info['total_duration_us'] = time.ticks_diff(time.ticks_us(), t_start)
        return None, None, None, timing_info

    # ── For capture modes ──
    if not sample_rate or sample_rate <= 0:
        raise ValueError("sample_rate must be > 0 for 'ping' and 'listen'")
    if not n_samples or n_samples <= 0:
        raise ValueError("n_samples must be > 0 for 'ping' and 'listen'")

    buf0 = array.array('H', (0 for _ in range(n_samples)))  # recv1
    buf1 = array.array('H', (0 for _ in range(n_samples)))  # recv2
    buf2 = array.array('H', (0 for _ in range(n_samples)))  # emitter monitor

    if mode == 'ping':
        emit_pulse(trigger_emitter)
        time.sleep_us(post_emit_settle_us)
        t_wait_start = time.ticks_us()
        while True:
            if adc_emit.read_u16() > threshold:
                timing_info['timeout'] = False
                break
            if time.ticks_diff(time.ticks_us(), t_wait_start) > timeout_us:
                timing_info['timeout'] = True
                break

        t_gate = time.ticks_us()
        timing_info['mode'] = 'ping'
        timing_info['threshold_detect_us'] = time.ticks_diff(t_gate, t_start)
        timing_info['threshold_wait_us']   = time.ticks_diff(t_gate, t_wait_start)

    else:  # mode == 'listen'
        t_gate = time.ticks_us()
        timing_info['mode'] = 'listen'
        timing_info['threshold_detect_us'] = 0

    # ── Timed sampling loop (absolute schedule) ──
    period_us = int(1_000_000 // int(sample_rate))
    if period_us <= 0: raise ValueError("sample_rate too high for microsecond scheduling")

    next_due = t_gate
    for i in range(n_samples):
        while time.ticks_diff(time.ticks_us(), next_due) < 0: pass
        # Read receivers and emitter monitor each sample tick
        buf0[i] = adc_emit.read_u16()
        buf1[i] = adc_recv1.read_u16()
        buf2[i] = adc_recv2.read_u16()
        next_due = time.ticks_add(next_due, period_us)

    t_end = time.ticks_us()
    timing_info['sampling_duration_us'] = time.ticks_diff(t_end, t_gate)
    timing_info['total_duration_us']    = time.ticks_diff(t_end, t_start)

    return buf0, buf1, buf2, timing_info
