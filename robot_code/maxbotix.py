from machine import Pin, ADC
import time
import array
import settings

class MaxBotix:
    def __init__(self):
        self.adc_emit  = ADC(settings.adc_emitter)
        self.adc_recv1 = ADC(settings.adc_recv1)
        self.adc_recv2 = ADC(settings.adc_recv2)

        self.trigger_emitter = Pin(settings.trigger_emitter, Pin.OUT)
        Pin(settings.trigger_recv1, Pin.OUT).value(0)
        Pin(settings.trigger_recv2, Pin.OUT).value(0)

        self.pulse_threshold = settings.pulse_threshold
        self.sample_rate = 10000
        self.n_samples = 500

        self.buf0 = None
        self.buf1 = None
        self.buf2 = None

    def measure(self, sample_rate=0, n_samples=0):
        if sample_rate == 0: sample_rate = self.sample_rate
        if n_samples == 0: n_samples = self.n_samples
        delay_us = int(1_000_000 / sample_rate)

        self.buf0 = array.array("H", [0] * n_samples)
        self.buf1 = array.array("H", [0] * n_samples)
        self.buf2 = array.array("H", [0] * n_samples)

        t0 = time.ticks_us()

        # Trigger emitter
        self.trigger_emitter.value(0)
        time.sleep_us(10)
        self.trigger_emitter.value(1)
        time.sleep_us(50)
        self.trigger_emitter.value(0)

        t1 = time.ticks_us()

        # Wait for threshold on emitter ADC
        while self.adc_emit.read_u16() < self.pulse_threshold: pass

        t2 = time.ticks_us()

        # Start sampling
        for i in range(n_samples):
            self.buf0[i] = self.adc_emit.read_u16()
            self.buf1[i] = self.adc_recv1.read_u16()
            self.buf2[i] = self.adc_recv2.read_u16()
            time.sleep_us(delay_us)

        t3 = time.ticks_us()

        print("Timing (μs):")
        print(f"  Triggering duration       : {time.tick