from machine import Pin, ADC
import time
import array
import settings  # Your custom settings file


def pulse():
    trigger_emitter = Pin(settings.trigger_emitter, Pin.OUT)
    # Mute the receiver sensors by pulling their trigger pins low
    Pin(settings.trigger_recv1, Pin.OUT).value(0)
    Pin(settings.trigger_recv2, Pin.OUT).value(0)
    # ───── Trigger Emitter ─────
    trigger_emitter.value(0)
    time.sleep_us(10)
    trigger_emitter.value(1)
    time.sleep_us(50)
    trigger_emitter.value(0)


def measure(sample_rate, n_samples, wait_for_emission=True):
    # ───── Use pin assignments from settings ─────
    trigger_emitter = Pin(settings.trigger_emitter, Pin.OUT)
    adc_emit  = ADC(settings.adc_emitter)
    adc_recv1 = ADC(settings.adc_recv1)
    adc_recv2 = ADC(settings.adc_recv2)

    # Mute the receiver sensors by pulling their trigger pins low
    Pin(settings.trigger_recv1, Pin.OUT).value(0)
    Pin(settings.trigger_recv2, Pin.OUT).value(0)

    delay_us = int(1_000_000 / sample_rate)
    pulse_threshold = settings.pulse_threshold

    # ───── Buffers ─────
    buf0 = array.array("H", [0] * n_samples)
    buf1 = array.array("H", [0] * n_samples)
    buf2 = array.array("H", [0] * n_samples)

    # ───── Timing t0: Before triggering ─────
    t0 = time.ticks_us()

    # ───── Trigger Emitter ─────
    trigger_emitter.value(0)
    time.sleep_us(10)
    trigger_emitter.value(1)
    time.sleep_us(50)
    trigger_emitter.value(0)

    # ───── Timing t1: After trigger ─────
    t1 = time.ticks_us()

    # ───── Wait for Emission Pulse ─────
    start_wait = time.ticks_us()
    timeout_us = 100_000  # Max wait time: 100 ms
    while True:
        if not wait_for_emission: break
        val = adc_emit.read_u16()
        if val > pulse_threshold: break
        if time.ticks_diff(time.ticks_us(), start_wait) > timeout_us:
            print("Timeout waiting for pulse")
            break

    # ───── Timing t2: After threshold detected ─────
    t2 = time.ticks_us()

    # ───── Sampling Loop ─────
    for i in range(n_samples):
        buf0[i] = adc_emit.read_u16()
        buf1[i] = adc_recv1.read_u16()
        buf2[i] = adc_recv2.read_u16()
        time.sleep_us(delay_us)

    # ───── Timing t3: After sampling ─────
    t3 = time.ticks_us()

    # ───── Timing Info ─────
    timing_info = {
        'triggering duration':       time.ticks_diff(t1, t0),
        'threshold detect (us)':     time.ticks_diff(t2, t1),
        'sampling duration (us)':    time.ticks_diff(t3, t2),
        'total duration (us)':       time.ticks_diff(t3, t0),
    }

    return buf0, buf1, buf2, timing_info

