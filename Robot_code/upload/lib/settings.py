import passwords
verbose = 2


ssid_list = {
    'batnet_robotlab': passwords.password0,
    'batnet': passwords.password1,
    'ZyXEL39940': passwords.password2
}

split_char = ','
end_char = '*'

tx_pin = 4
rx_pin = 5
en_pin = 27

pulse_threshold = 15000

adc_recv1 = 2
adc_recv2 = 3
trigger_recv1 = 7 #  used to pull down and avoid these triggering
trigger_recv2 = 24 # used to pull down and avoid these triggering

adc_emitter = 0 # used for emission detection
trigger_emitter = 22
