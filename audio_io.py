import concurrent.futures
import os
import platform
import time
import wave

import numpy as np
import pretty_midi
import pyaudio

import shared_utils

base_path = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.join(base_path, 'bin')
os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']

# default pyFluidsynth from PyPi doesn't work with fluidsynth 2
# a newer modified pyFluidsynth is required to support fluidsynth 2
# for more information, visit https://ksvi.mff.cuni.cz/~dingle/2019/prog_1/python_music.html
import fluidsynth


class AudioInput:
    # abstract class that should never be instantiated
    def __init__(self, config):
        self._chunk = config['perf_chunk']  # number of samples processed each iteration
        self._running = True  # flag
        self._proc = None
        self._first_run = True
        self._start_time = 0.0
        self._prev_time = 0.0

    def connect_to_proc(self, target):
        self._proc = target

    def loop(self):
        pass

    def kill(self):
        self._running = False

    @property
    def start_time(self):
        return self._start_time


class WaveFileInput(AudioInput):
    def __init__(self, config):
        super().__init__(config)

        self._wf = wave.open(config['perf_audio'], 'rb')
        if self._wf.getnchannels() != 1:
            raise Exception('unsupported channel number')
        self._samp_rate = self._wf.getframerate()
        self._samp_width = self._wf.getsampwidth()
        config['perf_sr'] = self._samp_rate  # reversely change config
        try:
            self._dtype = {
                1: np.int8,
                2: np.int16,
                4: np.int32,
                8: np.int64
            }[self._samp_width]
        except KeyError:
            raise Exception('unsupported sample width')
        self._denom = 2 ** (self._samp_width * 8 - 1)  # convert from integer to floating
        self._chunk_dur = self._chunk / self._samp_rate

        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(format=self._audio.get_format_from_width(self._wf.getsampwidth()),
                                        channels=self._wf.getnchannels(),
                                        rate=self._wf.getframerate(),
                                        output=True)

    def loop(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            while self._running:
                bytes_read = self._wf.readframes(self._chunk)
                # when use wave file as input, also write it to an output stream
                # this works as both preview and clock synchronization
                #
                # when use microphone as input,
                # every read action blocks until sufficient samples cumulated in buffer
                #
                # read from wave file return almost immediately,
                # however, write it to an output stream will block until all samples are played
                #
                # in order to align two inputs' behavior, when use wave file as input,
                # pass audio data to processor after playback finishes
                future = executor.submit(self._stream.write, bytes_read)  # play async
                future.result()  # wait for play end
                read_time = time.time()  # as accurate as possible, record time right after finishing reading
                if self._first_run:
                    self._first_run = False
                    # only execute in the first time
                    self._start_time = read_time - self._chunk_dur
                    self._prev_time = read_time - self._chunk_dur
                data = np.frombuffer(bytes_read, dtype=self._dtype)
                if len(data) != self._chunk:
                    # reaches file end, or remaining samples are insufficient to composite one chunk
                    self._running = False
                    continue
                data = np.true_divide(data, self._denom, dtype=np.float32)
                executor.submit(self._proc, read_time, self._prev_time, data, self)  # process async
                self._prev_time = read_time

        # some internal cleaning work
        self._wf.close()
        self._stream.stop_stream()
        self._stream.close()
        self._audio.terminate()


class MicrophoneInput(AudioInput):
    def __init__(self, config):
        super().__init__(config)

        self._samp_rate = config['perf_sr']
        self._dump = config['dump_mic']
        self._chunk_dur = self._chunk / self._samp_rate

        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=self._samp_rate,
                                        input=True,
                                        frames_per_buffer=self._chunk)

        if self._dump:
            self._dump_dir = config['name']
            self._dump_data = []

    def loop(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while self._running:
                bytes_read = self._stream.read(self._chunk)
                read_time = time.time()  # as accurate as possible, record time right after finishing reading
                if self._first_run:
                    self._first_run = False
                    # only execute in the first time
                    self._start_time = read_time - self._chunk_dur
                    self._prev_time = read_time - self._chunk_dur
                if self._dump:
                    self._dump_data.append(bytes_read)
                data = np.frombuffer(bytes_read, dtype=np.int16)
                data = np.true_divide(data, 2 ** 15, dtype=np.float32)
                executor.submit(self._proc, read_time, self._prev_time, data, self)  # process async
                self._prev_time = read_time

        # some internal cleaning work
        self._stream.stop_stream()
        self._stream.close()
        self._audio.terminate()

        if self._dump:
            shared_utils.check_dir('output', self._dump_dir)
            wave_file = wave.open('output/%s/mic.wav' % self._dump_dir, 'wb')
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(self._samp_rate)
            wave_file.writeframes(b''.join(self._dump_data))
            wave_file.close()


class AudioOutput:
    # abstract class that should never be instantiated
    def loop(self):
        pass

    def change_tempo_ratio(self, tempo_ratio):
        pass

    def kill(self):
        pass


class MIDIPlayer(AudioOutput):
    TICK_INT = 0.01  # second to wait between ticks, note that time.sleep() is not accurate
    STREAM_WAIT = 0.5  # second to wait before and after playing

    def __init__(self, config):
        self._midi_path = config['acco_midi']

        self._midi_events, self._midi_tempo = MIDIPlayer._load_midi(self._midi_path)  # tempo in BPS
        self._event_pos = 0  # all MIDI events must be executed in sequence, keep track of execution position
        self._cur_pos = 0  # current position in beat, updated every tick
        self._cur_tempo = self._midi_tempo  # current tempo in BPS, same as midi's tempo at first
        self._prev_tick = 0  # time of previous tick
        self._first_run = True  # flag
        self._running = True  # flag

        self._proc = None

        self._fs = fluidsynth.Synth()
        # automatically select driver based on platform
        pf = platform.platform()
        if pf.startswith('Windows'):
            self._fs.start(driver='dsound')
        elif pf.startswith('Darwin'):
            self._fs.start(driver='coreaudio')
        else:
            raise Exception('unsupported platform')
        sf = self._fs.sfload(r'resources/soundfont.sf2')
        self._fs.program_select(0, sf, 0, 0)
        # wait, sometimes stream does not open immediately
        time.sleep(MIDIPlayer.STREAM_WAIT)

    @staticmethod
    def _load_midi(midi_path):
        midi_file = pretty_midi.PrettyMIDI(midi_path)
        bpm = midi_file.get_tempo_changes()[1][0]
        bps = bpm / 60
        events = []
        for note in midi_file.instruments[0].notes:
            # time in beat, type (1=start and 0=end), pitch, velocity
            # to convert from second to beat, second * beat/second = beat
            # use time in beat because time in second should be updated each time tempo changes,
            # but time in beat doesn't
            events.append((note.start * bps, 1, note.pitch, note.velocity))
            events.append((note.end * bps, 0, note.pitch, 0))
        # sorted by time, when time is same, note off (0) goes first
        events = sorted(events, key=lambda x: (x[0], x[1]))
        return events, bps

    def connect_to_proc(self, proc):
        self._proc = proc

    def loop(self):
        # start looping
        while self._running:
            self._tick()
            time.sleep(MIDIPlayer.TICK_INT)
        # some internal cleaning work
        time.sleep(MIDIPlayer.STREAM_WAIT)
        self._fs.delete()

    def change_tempo_ratio(self, ratio):
        self._cur_tempo = self._midi_tempo * ratio

    def kill(self):
        self._running = False

    def _tick(self):
        # time.sleep() is inaccurate, always use time.time() for time measurement
        cur_tick = time.time()
        if self._first_run:
            elapsed_time = 0
            self._prev_tick = cur_tick
            self._first_run = False
        else:
            elapsed_time = cur_tick - self._prev_tick
            self._prev_tick = cur_tick
        # update current position
        # convert from second to beat
        self._cur_pos += self._cur_tempo * elapsed_time
        # execute MIDI events
        # may be more than one event
        while self._event_pos < len(self._midi_events) and self._midi_events[self._event_pos][0] <= self._cur_pos:
            ev = self._midi_events[self._event_pos]
            if ev[1] == 1:
                self._fs.noteon(0, ev[2], ev[3])
            elif ev[1] == 0:
                self._fs.noteoff(0, ev[2])
            self._event_pos += 1
        # execute external processor
        self._proc(cur_tick, self)
        # stop when all events finish
        if self._event_pos >= len(self._midi_events):
            self.kill()

    @property
    def current_position(self):
        return self._cur_pos


class WaveFileOutput(AudioOutput):
    # TODO: implement realtime tempo-variable wave file player
    pass


if __name__ == '__main__':
    test_config = {
        'name': 'audio_io_debug',
        'perf_audio': 'audio/audio3.wav',
        'perf_chunk': 1024,
        'perf_sr': 44100,
        'dump_mic': True,
        'acco_midi': 'midi/midi4_quick.mid'
    }

    # def test_proc(audio_time, prev_time, audio_data, audio_input):
    #     if audio_time > test_input.start_time + 15.0:
    #         audio_input.kill()
    #     la = int(abs(round(min(audio_data), 1) * 10))
    #     ra = int(abs(round(max(audio_data), 1) * 10))
    #     print('Time', format(audio_time - test_input.start_time, '8.5f'),
    #           '\tDuration', format(audio_time - prev_time, '8.5f'),
    #           '\tMin/Max', '[', '-' * (10 - la) + '#' * (la + ra) + '-' * (10 - ra), ']')
    #
    #
    # test_input = WaveFileInput(test_config)
    # test_input.connect_to_proc(test_proc)
    # test_input.loop()

    test_mp = MIDIPlayer(test_config)
    test_mp.connect_to_proc(lambda *args: None)
    test_mp.loop()
