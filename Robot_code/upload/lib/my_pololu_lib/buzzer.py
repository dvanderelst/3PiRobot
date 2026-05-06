# Modified from upstream Pololu library to free GP7 between beeps.
# Upstream acquired the PWM peripheral on GP7 at import time and held it
# for the program's lifetime; this collides with GP7's other use as the
# sonar recv1 trigger (settings.trigger_recv1). Here PWM is acquired on
# first use and released after each tune, returning GP7 to driven-low so
# the MB1360 recv1 sensor stays inhibited between beeps.
# See PATCHES.md.

from machine import Pin, PWM
import machine
import time

pwm = None
user_callback = lambda i: None
is_playing = False

volume_levels = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 64, 128]


def _acquire_pwm():
    global pwm
    if pwm is None:
        pwm = PWM(Pin(7, Pin.OUT))
    return pwm


def _release_pwm():
    # Tear down PWM and drive GP7 actively low so the sonar recv1 trigger
    # (which shares this pin) is held inhibited between beeps.
    global pwm
    if pwm is not None:
        pwm.duty_u16(0)
        pwm.deinit()
        pwm = None
    Pin(7, Pin.OUT, value=0)


class Buzzer:
    def __init__(self):
        # Ensure quiet state and GP7 driven low at construction.
        self.off()

    def is_playing(self):
        global is_playing
        return is_playing

    def set_callback(self, f):
        global user_callback
        user_callback = f

    def beep(self):
        p = _acquire_pwm()
        p.freq(440)
        p.duty_u16(32767)
        time.sleep_ms(100)
        _release_pwm()

    def on(self):
        # Continuous tone — caller must invoke off() to release GP7.
        p = _acquire_pwm()
        p.freq(200)
        p.duty_u16(32767)

    def off(self):
        global is_playing
        if is_playing:
            timer.deinit()
            is_playing = False
        _release_pwm()

    def play(self, notes):
        global is_playing
        try:
            self.play_in_background(notes)
            while is_playing:
                pass
        finally:
            self.off()

    def play_in_background(self, music):
        global i, note_count, timer, is_playing, volume_levels
        global frequencies, durations, volumes, notes, beats
        music = music.lower()

        if is_playing:
            timer.deinit()

        # Initialize the arrays that store the precomputed music
        # sequence.  We also make them accessible as instance
        # variables so you can monitor the state of the sequence.
        self.volumes = volumes = []
        self.durations = durations = []
        self.frequencies = frequencies = []
        self.beats = beats = [] # units of 1/20160 of a measure
        self.notes = notes = []

        x = 0

        octave = 4
        octave_boost = 0
        volume = 15
        staccato = False
        tempo = 120
        default_duration = 4
        elapsed_beats = 0

        music_len = len(music)
        while x < music_len:
            c = music[x]
            x += 1
            num_string = ""

            accidentals = 0
            if x < music_len and (music[x] == '+' or music[x] == '#'):
                accidentals += 1
                x += 1
            elif x < music_len and music[x] == '-':
                accidentals -= 1
                x += 1

            while x < music_len and music[x].isdigit():
                num_string += music[x]
                x += 1
            num = int(num_string) if num_string != "" else 0

            note = octave * 12
            if "c" == c:
                note += 0
            elif "d" == c:
                note += 2
            elif "e" == c:
                note += 4
            elif "f" == c:
                note += 5
            elif "g" == c:
                note += 7
            elif "a" == c:
                note += 9
            elif "b" == c:
                note += 11
            elif "r" == c:
                note = 0
            elif ">" == c:
                octave_boost += 1
                continue
            elif "<" == c:
                octave_boost -= 1
                continue
            elif "t" == c:
                tempo = num
                continue
            elif "v" == c:
                volume = min(num, 15)
                continue
            elif "o" == c:
                octave = num
                continue
            elif "l" == c:
                default_duration = min(num, 2000)
                continue
            elif "m" == c:
                staccato = x < music_len and music[x] == 's'
                x += 1
                continue
            elif "!" == c:
                octave = 4
                volume = 15
                staccato = False
                tempo = 120
                default_duration = 4
                octave_boost = 0
                continue
            else:
                continue

            note += octave_boost*12
            note += accidentals

            duration_type = 1
            if num > 0 and num <= 2000:
                duration_type = num
            else:
                duration_type = default_duration
            duration = 60000/tempo/(duration_type/4)

            # Compute integer duration in units of 1/20160 of a
            # quarter note, for keeping the beat.
            duration_beats = 20160*4//duration_type

            if x < music_len and music[x] == '.':
                duration += duration/2
                duration_beats += duration_beats//2
                x += 1

            if note == 0:
                frequencies.append(0)
                volumes.append(0)
                notes.append(0)
            else:
                freq = round(440 * 2**((note - 57)/12))
                frequencies.append(freq)
                volumes.append(volume_levels[volume])
                notes.append(note)
            durations.append(round(duration/2 if staccato else duration))

            beats.append(elapsed_beats)
            elapsed_beats += (duration_beats // 2 if staccato else duration_beats)

            if staccato:
                frequencies.append(0)
                durations.append(round(duration / 2))
                volumes.append(0)
                notes.append(0)
                beats.append(elapsed_beats)
                elapsed_beats += duration_beats // 2

            octave_boost = 0
        # end of loop

        _acquire_pwm()
        is_playing = True
        i = 0
        note_count = len(frequencies)
        timer = machine.Timer()
        timer.init(period=1, mode=machine.Timer.ONE_SHOT, callback=callback)

def callback(t):
    global pwm, i, frequencies, volumes, durations, note_count, is_playing

    if i >= note_count:
        # End of tune: release GP7 back to driven-low for sonar recv1.
        _release_pwm()
        is_playing = False
        return

    if frequencies[i]:
        pwm.freq(frequencies[i])
    pwm.duty_u16(volumes[i]*256)
    user_callback(i)
    t.init(period=durations[i], mode=machine.Timer.ONE_SHOT, callback=callback)
    i += 1
