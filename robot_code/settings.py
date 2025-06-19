import passwords

verbose = 2
ssid_index = 1  # Choose which known SSID to connect to


ssid_list = [
    ('batnet', passwords.password0),
    ('ZyXEL39940', passwords.password1),
    # Add more if needed
]

split_char = ','
end_char = '*'

tx_pin = 4
rx_pin = 5
en_pin = 27

pulse_threshold = 15000# 17000

adc_recv1 = 2
adc_recv2 = 3
trigger_recv1 = 7 #  used to pull down and avoid these triggering
trigger_recv2 = 24 # used to pull down and avoid these triggering

adc_emitter = 0 # used for emission detection
trigger_emitter = 22