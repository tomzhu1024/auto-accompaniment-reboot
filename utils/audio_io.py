import concurrent.futures
import os
import platform
import time
import wave

import numpy as np
import pretty_midi
import pyaudio

from . import shared_utils


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
                if self._first_run:
                    self._first_run = False
                    # only execute in the first time
                    read_time = time.time()
                    self._start_time = read_time - self._chunk_dur
                    self._prev_time = read_time - self._chunk_dur
                else:
                    read_time = self._prev_time + self._chunk_dur
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
        self._dump = config['dump']
        self._chunk_dur = self._chunk / self._samp_rate

        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=self._samp_rate,
                                        input=True,
                                        frames_per_buffer=self._chunk)
        if self._dump:
            self._dump_data = []

    def loop(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while self._running:
                bytes_read = self._stream.read(self._chunk)
                read_time = time.time()
                if self._first_run:
                    self._first_run = False
                    # only execute in the first time
                    self._start_time = read_time - self._chunk_dur
                    self._prev_time = read_time - self._chunk_dur
                if self._dump:
                    self._dump_data.append(bytes_read)
                data = np.frombuffer(bytes_read, dtype=np.int16)
                # sampling width in microphone input is fixed to 16bit
                data = np.true_divide(data, 2 ** 15, dtype=np.float32)
                # self._proc(read_time, self._prev_time, data, self)  # process sync
                executor.submit(self._proc, read_time, self._prev_time, data, self)
                self._prev_time = read_time

        # some internal cleaning work
        self._stream.stop_stream()
        self._stream.close()
        self._audio.terminate()

    def save_to_file(self, path):
        if self._dump:
            wave_file = wave.open(path, 'wb')
            wave_file.setnchannels(1)  # mono
            wave_file.setsampwidth(2)  # 2 bytes is 16 bits
            wave_file.setframerate(self._samp_rate)
            wave_file.writeframes(b''.join(self._dump_data))
            wave_file.close()


class MIDISynthesizer:
    def note_on(self, pitch, velocity):
        pass

    def note_off(self, pitch):
        pass

    def close(self):
        pass


class VirtualSynthesizer(MIDISynthesizer):
    def __init__(self):
        # get platform
        pf = platform.platform()
        if pf.startswith('Windows'):
            # for Windows, add local binary directory to path
            base_path = os.path.dirname(os.path.abspath(__file__))
            bin_path = os.path.join(base_path, '../bin')
            os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']

        # https://ksvi.mff.cuni.cz/~dingle/2019/prog_1/python_music.html
        import fluidsynth

        self._fs = fluidsynth.Synth()
        if pf.startswith('Windows'):
            self._fs.start(driver='dsound')
        elif pf.startswith('Darwin'):
            self._fs.start(driver='coreaudio')
        else:
            raise Exception('unsupported platform')
        sf = self._fs.sfload(r'resources/soundfont.sf2')
        self._fs.program_select(0, sf, 0, 0)

    def note_on(self, pitch, velocity):
        self._fs.noteon(0, pitch, velocity)

    def note_off(self, pitch):
        self._fs.noteoff(0, pitch)

    def close(self):
        self._fs.delete()


class ExternalSynthesizer(MIDISynthesizer):
    def __init__(self, config):
        import rtmidi

        self._midi_port = rtmidi.MidiOut()
        available_ports = self._midi_port.get_ports()
        print('Available ports:', available_ports)

        port_index = -1
        for i in range(len(available_ports)):
            if available_ports[i].startswith(config['acco_device']):
                port_index = i
        if available_ports and port_index != -1:
            self._midi_port.open_port(port_index)
            print('Opened port:', available_ports[port_index])
        else:
            raise Exception('could not find port: %s' % config['acco_device'])

    def note_on(self, pitch, velocity):
        self._midi_port.send_message([0x90, pitch, velocity])

    def note_off(self, pitch):
        self._midi_port.send_message([0x80, pitch, 0])

    def close(self):
        del self._midi_port


class AudioOutput:
    def __init__(self):
        self._running = True  # flag
        self._proc = None

    def connect_to_proc(self, proc):
        self._proc = proc

    def loop(self):
        pass

    def kill(self):
        self._running = False

    def change_tempo_ratio(self, ratio):
        pass

    @property
    def current_time(self):
        return

    @property
    def total_time(self):
        return


class MIDIPlayer(AudioOutput):
    TICK_INT = 0.02  # second to wait between ticks, note that time.sleep() is not accurate

    def __init__(self, midi_path, output: MIDISynthesizer):
        super().__init__()

        self._midi_events, self._midi_tempo = MIDIPlayer._load_midi(midi_path)  # tempo in BPS
        self._event_pos = 0  # all MIDI events must be executed in sequence, keep track of execution position
        self._cur_pos = 0  # current position in beat, updated every tick
        self._cur_tempo = self._midi_tempo  # current tempo in BPS, same as MIDI's tempo at first
        self._prev_tick = 0  # time of previous tick
        self._first_run = True  # flag

        self._output = output

    @staticmethod
    def _load_midi(midi_path):
        midi_file = pretty_midi.PrettyMIDI(midi_path)
        bpm = shared_utils.average(midi_file.get_tempo_changes()[1])
        bps = bpm / 60
        events = []
        for note in midi_file.instruments[0].notes:
            # time in beat, type (1=start and 0=end), pitch, velocity
            # to convert from second to beat, second * beat/second = beat
            # use time in beat because time in second should be updated each time tempo changes,
            # but time in beat doesn't
            events.append((note.start * bps, 1, note.pitch, note.velocity))
            events.append((note.end * bps, 0, note.pitch, 0))
        # sorted by time, when time is same, note off event (the second number is 0 rather than 1) goes first
        #
        #   Note 1  |======|
        #   Note 2         |====|
        #     for example, in the above case, two notes have a same pitch, this method avoids merging two notes into one
        #
        #   Note 1  |===========|
        #   Note 2      |===|
        #   sure enough, this method still has some problems with the case above,
        #   but we assume that one key cannot be pressed down twice without releasing it,
        #   so such cases doesn't need to be considered
        events = sorted(events, key=lambda x: (x[0], x[1]))
        return events, bps

    def loop(self):
        # start looping
        while self._running:
            self._tick()
            time.sleep(MIDIPlayer.TICK_INT)
        # some internal cleaning work
        self._output.close()

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
        # assume that all MIDI events are in order
        # there may be more than one event to execute
        while self._event_pos < len(self._midi_events) and self._midi_events[self._event_pos][0] <= self._cur_pos:
            ev = self._midi_events[self._event_pos]
            if ev[1] == 1:
                self._output.note_on(ev[2], ev[3])
            elif ev[1] == 0:
                self._output.note_off(ev[2])
            self._event_pos += 1
        # execute external processor
        self._proc(cur_tick, self)
        # stop when all events finish
        if self._event_pos >= len(self._midi_events):
            self.kill()

    def change_tempo_ratio(self, ratio):
        self._cur_tempo = self._midi_tempo * ratio

    @property
    def current_time(self):
        return self._cur_pos / self._midi_tempo

    @property
    def total_time(self):
        return self._midi_events[-1][0] / self._midi_tempo
