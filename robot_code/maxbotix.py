from machine import Pin, ADC, Timer
import leds
import array
import time

class MaxBotix:
    def __init__(self):
        self.trigger = Pin(24, Pin.OUT)
        self.adc_emit = ADC(1)
        self.adc_recv1 = ADC(1)
        self.adc_recv2 = ADC(2)

        self.wait_method = "threshold"
        self.pulse_threshold = 15000
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
        
        self.leds = leds.LEDs()

    def _sample_callback(self, timer):
        if self._index < len(self.buf1):
            self.buf1[self._index] = self.adc_recv1.read_u16()
            self.buf2[self._index] = self.adc_recv2.read_u16()
            self._index += 1
        else:
            timer.deinit()
            self._done = True

    def _wait_for_pulse(self):
        start = time.ticks_us()
        while time.ticks_diff(time.ticks_us(), start) < self.max_delay_us:
            if self.adc_emit.read_u16() > self.pulse_threshold:
                return True
        return False

    def measure(self):
        n_samples = self.n_samples
        sample_rate = self.sample_rate
        
        self.leds.set_all('orange')

        self.buf1 = array.array("H", [0] * n_samples)
        self.buf2 = array.array("H", [0] * n_samples)
        self._index = 0
        self._done = False

        self.trigger.value(1)
        time.sleep_ms(30)
        self.trigger.value(0)

        if self.wait_method == "threshold":
            self._wait_for_pulse()
        elif self.wait_method == "fixed":
            time.sleep_us(self.fixed_delay_us)
        else:
            raise ValueError("Invalid wait_method.")

        self._timer.init(freq=sample_rate, mode=Timer.PERIODIC, callback=self._sample_callback)

        while not self._done:
            pass
        
        self.leds.set_all('off')
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

