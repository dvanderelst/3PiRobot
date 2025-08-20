from machine import Pin, ADC
import time
import array
import settings  # Your custom settings file

# ───── Use pin assignments from settings ─────
trigger_emitter = Pin(settings.trigger_emitter, Pin.OUT)
adc_emit  = ADC(settings.adc_emitter)
adc_recv1 = ADC(settings.adc_recv1)
adc_recv2 = ADC(settings.adc_recv2)

# Mute the receiver sensors by pulling their trigger pins low
Pin(settings.trigger_recv1, Pin.OUT).value(0)
Pin(settings.trigger_recv2, Pin.OUT).value(0)

# ───── Sampling Parameters ─────
n_samples = 500
sample_rate = 10000  # 10kHz → 100 μs per sample
delay_us = int(1_000_000 / sample_rate)
pulse_threshold = settings.pulse_threshold  # <- pulled from settings

# ───── Buffers ─────
buf0 = array.array("H", [0] * n_samples)  # emitter AE
buf1 = array.array("H", [0] * n_samples)  # receiver 1 AE
buf2 = array.array("H", [0] * n_samples)  # receiver 2 AE

# ───── Trigger Emitter ─────
trigger_emitter.value(0)
time.sleep_us(10)
trigger_emitter.value(1)
time.sleep_us(50)
trigger_emitter.value(0)

# ───── Wait for Emission Pulse ─────
start_wait = time.ticks_us()
timeout_us = 100000  # Max wait time: 100 ms

while True:
    val = adc_emit.read_u16()
    if val > pulse_threshold:
        break
    if time.ticks_diff(time.ticks_us(), start_wait) > timeout_us:
        print("Timeout waiting for pulse")
        break

# ───── Sampling Loop ─────
for i in range(n_samples):
    buf0[i] = adc_emit.read_u16()
    buf1[i] = adc_recv1.read_u16()
    buf2[i] = adc_recv2.read_u16()
    time.sleep_us(delay_us)

# ───── Print First Samples ─────
print("Index\tEmitter\tRecv1\tRecv2")
for i in range(n_samples):
    print(f"{i}\t{buf0[i]}\t{buf1[i]}\t{buf2[i]}")
