from machine import Pin, ADC, Timer
import leds
import array
import time
import settings

class MaxBotix:
    def __init__(self):
        self.verbose = settings.verbose
        self.trigger = Pin(settings.trigger_emitter, Pin.OUT)
        self.recv1_enable = Pin(settings.trigger_recv1, Pin.OUT)
        self.recv2_enable = Pin(settings.trigger_recv2, Pin.OUT)
        
        
        self.adc_emit = ADC(settings.adc_emitter)
        self.adc_recv1 = ADC(settings.adc_recv1)
        self.adc_recv2 = ADC(settings.adc_recv2)

        self.wait_method = "threshold"
        self.pulse_threshold = settings.pulse_threshold
        self.fixed_delay_us = 10
        self.max_delay_us = 1000000

        self.sample_rate = 10000
        self.n_samples = 500
        
        self.vref = 3.3

        self.buf1 = None
        self.buf2 = None
        self._index = 0
        self._done = False
        self._timer = Timer()

        self.trigger.value(0)
        self.recv1_enable.value(0)
        self.recv2_enable.value(0)
        
        if self.verbose: print("[SNR] Initialized")

    def _sample_callback(self, timer):
        if self._index < len(self.buf1):
            self.buf1[self._index] = self.adc_recv1.read_u16()
            self.buf2[self._index] = self.adc_recv2.read_u16()
            self._index += 1
        else:
            timer.deinit()
            self._done = True

    def _wait_for_pulse(self):
        current_max = 0
        start = time.ticks_us()
        while time.ticks_diff(time.ticks_us(), start) < self.max_delay_us:
            value = self.adc_emit.read_u16()
            current_max = max(value, current_max)
            if  value > self.pulse_threshold: return True, current_max
        return False, current_max

    def measure(self, sample_rate=0, n_samples=0):
        if sample_rate == 0: sample_rate = self.sample_rate
        if n_samples == 0: n_samples = self.n_samples
        
        #self.leds.set_all('orange')

        self.buf1 = array.array("H", [0] * n_samples)
        self.buf2 = array.array("H", [0] * n_samples)
        self._index = 0
        self._done = False

        self.trigger.value(1)
        time.sleep_ms(30)
        self.trigger.value(0)
        exceeded = False
        max_value = 0
        if self.wait_method == "threshold":
            exceeded, max_value = self._wait_for_pulse()
        elif self.wait_method == "fixed":
            time.sleep_us(self.fixed_delay_us)
        else:
            raise ValueError("Invalid wait_method.")
        
        if self.verbose: print(f'[SNR] wait method {self.wait_method}, exceeded: {exceeded}, {max_value}')

        self._timer.init(freq=sample_rate, mode=Timer.PERIODIC, callback=self._sample_callback)

        while not self._done:
            pass
        
        #self.leds.set_all('off')
        return self.buf1, self.buf2

    def get_voltages(self):
        if self.buf1 is None or self.buf2 is None:
            return [], []
        scale = lambda buf: [v * self.vref / 65535 for v in buf]
        return scale(self.buf1), scale(self.buf2)

    def print_summary(self, step=100):
        if self.buf1 is None or self.buf2 is None:
            print("No data available.")
            return
        for i in range(0, len(self.buf1), step):
            v1 = self.buf1[i] * self.vref / 65535
            v2 = self.buf2[i] * self.vref / 65535
            print(f"Sample {i}: recv1 = {v1:.2f} V, recv2 = {v2:.2f} V")
    
    @property
    def max_distance(self):
        speed = 331
        distance = self.duration * speed / 2
        return distance
    
    @property
    def duration(self):
        duration = self.n_samples / self.sample_rate
        return duration

