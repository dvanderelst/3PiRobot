from machine import Pin, ADC, Timer
import array
import time

# --- CONFIGURATION ---
TRIGGER_PIN = 24     # GPIO24 (connected to sensor pin 4 - RX)
ADC_CHANNEL = 1      # ADC1 reads from GPIO27 (connected to sensor pin 2 - envelope)
N_SAMPLES = 50     # Number of samples
SAMPLE_RATE = 10000  # Hz

# --- SETUP ---
trigger = Pin(TRIGGER_PIN, Pin.OUT)
trigger.value(0)  # Ensure it's low initially
adc = ADC(ADC_CHANNEL)
buf = array.array("H", [0] * N_SAMPLES)

index = 0
done = False

# --- CALLBACK FUNCTION ---
def sample_callback(timer):
    global index, done
    if index < N_SAMPLES:
        buf[index] = adc.read_u16()
        index += 1
    else:
        timer.deinit()
        done = True

# --- TRIGGER SENSOR ---
def trigger_sensor():
    trigger.value(1)
    time.sleep_ms(30)  # Hold high for 30 ms
    trigger.value(0)

# --- MAIN ---
print("Triggering sensor...")
trigger_sensor()

print("Sampling analog envelope...")
t = Timer()
t.init(freq=SAMPLE_RATE, mode=Timer.PERIODIC, callback=sample_callback)

while not done:
    pass

print("Done! Sampled", N_SAMPLES, "points at", SAMPLE_RATE, "Hz")

# --- OPTIONAL: Print or Process ---
#for i in range(0, N_SAMPLES, 100):  # Print every 100th sample
#    voltage = buf[i] * 3.3 / 65535
#    print(f"Sample {i}: {voltage:.2f} V")

for s in buf: print(s)
