from machine import Pin, ADC
import time, array, settings

class MaxBotix:
    def __init__(self):
        self.trigger_emitter = Pin(settings.trigger_emitter, Pin.OUT)
        self.trigger_recv1 = Pin(settings.trigger_recv1, Pin.OUT)
        self.trigger_recv2 = Pin(settings.trigger_recv2, Pin.OUT)
        
        self.adc_emit  = ADC(settings.adc_emitter)
        self.adc_recv1 = ADC(settings.adc_recv1)
        self.adc_recv2 = ADC(settings.adc_recv2)
    
        self.pulse_threshold = settings.pulse_threshold
        self.sample_rate     = 10000
        self.n_samples       = 500
        
        self.trigger_emitter.value(0)
        self.trigger_recv1.value(0)
        self.trigger_recv2.value(0)
        
        self.buf0 = None
        self.buf1 = None
        self.buf2 = None
        self.timing_info = {}

    
    def run(self, value):
        self.trigger_emitter.value(value)
        self.trigger_recv1.value(value)
        self.trigger_recv2.value(value)
        
    def measure(self, sample_rate=0, n_samples=0):
        # apply defaults
        if sample_rate == 0: sample_rate = self.sample_rate
        if n_samples   == 0: n_samples   = self.n_samples
        delay_us = int(1_000_000 / sample_rate)

        # allocate buffers
        self.buf0 = array.array("H", [0] * n_samples)
        self.buf1 = array.array("H", [0] * n_samples)
        self.buf2 = array.array("H", [0] * n_samples)
        
        
        t0 = time.ticks_us()
        
  
        # fire the emitter
        self.trigger_emitter.value(0)
        time.sleep_us(10)
        self.trigger_emitter.value(1)
        time.sleep_us(50)
        self.trigger_emitter.value(0)
        t1 = time.ticks_us()
        
        print('wait')
        while True:
            value = self.adc_emit.read_u16()
            if value > self.pulse_threshold: break
        t2 = time.ticks_us()

        # sample all three channels
        for i in range(n_samples):
            self.buf0[i] = self.adc_emit.read_u16()
            self.buf1[i] = self.adc_recv1.read_u16()
            self.buf2[i] = self.adc_recv2.read_u16()
            time.sleep_us(delay_us)
        
        t3 = time.ticks_us()
        
        self.buf0 = list(self.buf0)
        self.buf1 = list(self.buf1)
        self.buf2 = list(self.buf2)

        # package timings
        self.timing_info = {
            'triggering duration'       : time.ticks_diff(t1, t0),
            'threshold detect (us)'     : time.ticks_diff(t2, t1),
            'sampling duration (us)'    : time.ticks_diff(t3, t2),
            'total duration (us)'       : time.ticks_diff(t3, t0),
        }








# from machine import Pin, ADC
# import time
# import array
# import settings
# 
# class MaxBotix:
#     def __init__(self):
#         self.adc_emit  = ADC(settings.adc_emitter)
#         self.adc_recv1 = ADC(settings.adc_recv1)
#         self.adc_recv2 = ADC(settings.adc_recv2)
# 
#         self.trigger_emitter = Pin(settings.trigger_emitter, Pin.OUT)
#         Pin(settings.trigger_recv1, Pin.OUT).value(0)
#         Pin(settings.trigger_recv2, Pin.OUT).value(0)
# 
#         self.pulse_threshold = settings.pulse_threshold
#         self.sample_rate = 10000
#         self.n_samples = 500
# 
#         self.buf0 = None
#         self.buf1 = None
#         self.buf2 = None
# 
#     def measure(self, sample_rate=0, n_samples=0):
#         if sample_rate == 0: sample_rate = self.sample_rate
#         if n_samples == 0: n_samples = self.n_samples
#         delay_us = int(1_000_000 / sample_rate)
# 
#         self.buf0 = array.array("H", [0] * n_samples)
#         self.buf1 = array.array("H", [0] * n_samples)
#         self.buf2 = array.array("H", [0] * n_samples)
# 
#         t0 = time.ticks_us()
# 
#         # Trigger emitter
#         self.trigger_emitter.value(0)
#         time.sleep_us(10)
#         self.trigger_emitter.value(1)
#         time.sleep_us(50)
#         self.trigger_emitter.value(0)
# 
#         t1 = time.ticks_us()
# 
#         # Wait for threshold on emitter ADC
#         while self.adc_emit.read_u16() < self.pulse_threshold: pass
# 
#         t2 = time.ticks_us()
# 
#         # Start sampling
#         for i in range(n_samples):
#             self.buf0[i] = self.adc_emit.read_u16()
#             self.buf1[i] = self.adc_recv1.read_u16()
#             self.buf2[i] = self.adc_recv2.read_u16()
#             time.sleep_us(delay_us)
# 
#         t3 = time.ticks_us()
#         
#         timing_info = {}
#         timing_info['triggering duration'] = time.ticks_diff(t1, t0)
#         timing_info['threshold detection delay'] = time.ticks_diff(t2, t1)
#         timing_info['sampling duration'] = time.ticks_diff(t3, t2)
#         timing_info['total duration'] = time.ticks_diff(t3, t0)
# 
#         return self.buf0, self.buf1, self.buf2, timing_info
# 