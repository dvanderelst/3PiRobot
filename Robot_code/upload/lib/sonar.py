from machine import Pin, ADC
import time
import array
import settings
import struct

def now():
    ticks = time.ticks_us()
    return ticks

def time_since(previous_time):
    current_time = time.ticks_us()
    difference = time.ticks_diff(current_time , previous_time)
    return difference

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
        
        self.tgr_emit = Pin(settings.trigger_emitter, Pin.OUT)
        self.tgr_recv1 = Pin(settings.trigger_recv1, Pin.OUT)
        self.tgr_recv2 = Pin(settings.trigger_recv2, Pin.OUT)
        
        self.tgr_emit.value(0)
        self.tgr_recv1.value(0)
        self.tgr_recv2.value(0)
        
        self.timeout_us = 100_000
        self.post_emit_settle_us = 20
        
    def emit(self):
        self.tgr_recv1.value(0)
        self.tgr_recv2.value(0)
        self.tgr_emit.value(1)
        time.sleep_us(75)
        self.tgr_emit.value(0)
            
    def create_buffers(self):
        n_samples = self.n_samples
        self.buf0 = array.array('H', (0 for _ in range(n_samples)))  
        self.buf1 = array.array('H', (0 for _ in range(n_samples)))  
        self.buf2 = array.array('H', (0 for _ in range(n_samples)))
        
    def update_buffers(self, n_samples):
        if n_samples is None: return
        if not n_samples == self.n_samples:
            self.n_samples = n_samples
            self.create_buffers()
            
    def fill_buffers(self, sample_rate):
        sample_period = int(1_000_000 // int(sample_rate)) #in usecs
        n_samples = self.n_samples
        next_sample_due = now()
        for i in range(n_samples):
            while time.ticks_diff(time.ticks_us(), next_sample_due) < 0: pass
            self.buf0[i] = self.adc_emit.read_u16()
            self.buf1[i] = self.adc_recv1.read_u16()
            self.buf2[i] = self.adc_recv2.read_u16()
            next_sample_due = time.ticks_add(next_sample_due, sample_period)

    def wait_for_emission(self):
        threshold = settings.pulse_threshold
        timeout_us = self.timeout_us
        start_of_wait = now()
        while True:
            if self.adc_emit.read_u16() > threshold: return True
            if time_since(start_of_wait) > timeout_us: return False
            time.sleep_us(1)
     
    def acquire(self, mode, sample_rate=None, n_samples=None):
        mode = mode.lower()
        start_acquire = now()
        timing_info = {}
        # ── Mode: pulse ──
        if mode == 'pulse':
            self.emit()
            timing_info['mode'] = 'pulse'
            timing_info['total_duration_us'] = time_since(start_acquire)
            return None, None, None, timing_info
        
        # ── Modes: pulse or ping──
        self.update_buffers(n_samples)

        if mode == 'ping':
            timing_info['emission_delay_us'] = time_since(start_acquire)
            self.emit()
            time.sleep_us(self.post_emit_settle_us)
            emission_detected = self.wait_for_emission()
            timing_info['emission_detected'] = emission_detected
        
        if mode in ['ping', 'listen']: self.fill_buffers(sample_rate)
        timing_info['total_duration_us'] = time_since(start_acquire)
        timing_info['mode'] = mode
        return self.buf0, self.buf1, self.buf2, timing_info

