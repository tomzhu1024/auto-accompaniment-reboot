import math
import multiprocessing
import time

import numpy as np
import pretty_midi
import statsmodels.api as sm

import audio_io
import shared_utils
import signal_processing


def normalize(array):
    return np.true_divide(array, np.sum(array))


def norm_pdf(x, mean, sd=1):
    var = sd ** 2
    denom = (2 * math.pi * var) ** 0.5
    num = math.exp(-(x - mean) ** 2 / (2 * var))
    return num / denom


def similarity(left, right):
    return (min(left, right) + 1e-6) / (max(left, right) + 1e-6)


def gate_mask(array, center, half_size):
    left_index = max(0, center - half_size)
    right_index = min(center + half_size, len(array))
    array[:left_index] = 0
    array[right_index:] = 0
    return array


class ScoreFollower:
    # sax_ means Score AXis
    # a_ means Audio
    RESOLUTION = 0.01

    def __init__(self, config):
        self._midi_path = config['score_midi']
        self._score_tempo, self._sax_time, self._sax_pitch, self._sax_onset, self._sax_length = \
            ScoreFollower._load_score(self._midi_path, ScoreFollower.RESOLUTION)
        self._f_source = np.zeros(self._sax_length)
        self._f_source[0] = 1.0
        self._cur_pos = 0
        self._estimated_tempo = self._score_tempo
        self._prev_report_pos = 0
        self._prev_tempo_pos = 0
        self._prev_tempo_time = 0
        self._first_run = True
        self._no_move = False

        self._tempo_ub = 1.3 * self._score_tempo
        self._tempo_lb = 0.7 * self._score_tempo

        if config['perf_mode'] == 0:
            self._audio_input = audio_io.WaveFileInput(config)
        elif config['perf_mode'] == 1:
            self._audio_input = audio_io.MicrophoneInput(config)
        self._audio_input.connect_to_proc(self._proc)
        self._pitch_proc = signal_processing.PitchProcessor(config)
        self._onset_proc = signal_processing.OnsetProcessor(config)

        self._auto_acco = AutoAccompaniment(config, self._score_tempo)

        # dump information for debug purpose
        self._dump = config['dump_sf']
        if self._dump:
            self._dump_dir = config['name']
            # dump midi
            self._d_midi_ref = pretty_midi.PrettyMIDI(self._midi_path)
            self._d_midi_new = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))
            self._d_midi_new.instruments.append(piano)
            self._d_midi_pos = 0
            # dump plot
            self._d_plot_ij = []
            self._d_plot_i = []
            self._d_plot_v = []
            self._d_plot_post = []  # posterior
            # dump performance info
            self._d_perf = []
            self._perf_start = 0

    @staticmethod
    def _load_score(midi_path, resolution):
        midi_file = pretty_midi.PrettyMIDI(midi_path)
        instrument = midi_file.instruments[0]
        score_tempo = midi_file.get_tempo_changes()[1][0]

        sax_length = int(math.ceil(max([note.end for note in instrument.notes]) / resolution)) + 1  # include 0
        sax_time = np.arange(0, sax_length * resolution, resolution)
        sax_pitch = np.zeros(sax_length)
        sax_onset = np.zeros(sax_length)
        pitches = np.full(sax_length, signal_processing.PitchProcessorCore.NO_PITCH)
        for note in instrument.notes:
            start = int(math.ceil(note.start / resolution))
            end = int(math.ceil(note.end / resolution)) + 1  # end will never go out of range
            pitches[start: end] = note.pitch
            sax_onset[start] = 1
        score_pitch_proc = signal_processing.PitchProcessorCore()
        for i in range(sax_length):
            sax_pitch[i] = score_pitch_proc(pitches[i])
        return score_tempo, sax_time, sax_pitch, sax_onset, sax_length

    def loop(self):
        self._auto_acco.loop()  # non-blocking, start new process for playing
        print('\t\tAuto acco proc started')
        self._audio_input.loop()  # blocking, block main process for reading, use separate thread for processing
        # some cleaning work after loop
        if self._dump:
            shared_utils.check_dir('output', self._dump_dir)
            self._d_midi_new.write('output/%s/sf.mid' % self._dump_dir)
            np.save('output/%s/ij.npy' % self._dump_dir, self._d_plot_ij)
            np.save('output/%s/i.npy' % self._dump_dir, self._d_plot_i)
            np.save('output/%s/v.npy' % self._dump_dir, self._d_plot_v)
            np.save('output/%s/post.npy' % self._dump_dir, self._d_plot_post)
            np.save('output/%s/perf.npy' % self._dump_dir, self._d_perf)

    def _proc(self, a_time, prev_a_time, a_data, a_input: audio_io.AudioInput):
        # called by audio input, execute in thread pool
        if self._dump:
            self._perf_start = time.perf_counter()

        if self._first_run:
            self._first_run = False
            # only execute in the first time
            self._prev_tempo_time = a_input.start_time

        a_pitch = self._pitch_proc(a_data)
        a_onset = self._onset_proc(a_data)

        # w-values fully depends on confidence of last estimation
        # if high-confidence, pitch weights more
        # if low-confidence, onset weights more
        # f_source and cur_pos remains unchanged here
        if self._f_source[self._cur_pos] > 0.1:
            w = (0.95, 0.05, 0.5)
        else:
            w = (0.7, 0.3, 0.3)

        # prior
        if self._no_move:
            f_i_j_given_d = None
            f_i_given_d = self._f_source
        else:
            f_i_j_given_d = ScoreFollower._compute_f_i_j_given_d(time_axis=self._sax_time,
                                                                 d=a_time - prev_a_time,
                                                                 score_tempo=self._score_tempo,
                                                                 estimated_tempo=self._estimated_tempo)
            f_i_given_d = ScoreFollower._compute_f_i_given_d(f_source=self._f_source,
                                                             f_i_j_given_d=f_i_j_given_d,
                                                             cur_pos=self._cur_pos,
                                                             axis_length=self._sax_length)
        # update position
        self._cur_pos = np.argmax(f_i_given_d)

        # observation
        f_v_given_i = ScoreFollower._compute_f_v_given_i(pitch_axis=self._sax_pitch,
                                                         onset_axis=self._sax_onset,
                                                         cur_pos=self._cur_pos,
                                                         axis_length=self._sax_length,
                                                         audio_pitch=a_pitch,
                                                         audio_onset=a_onset,
                                                         pitch_proc=self._pitch_proc,
                                                         w=w)
        # posterior
        self._f_source = f_i_given_d * f_v_given_i
        self._f_source = gate_mask(self._f_source, center=self._cur_pos, half_size=50)
        self._f_source = normalize(self._f_source)
        # update position again
        self._cur_pos = np.argmax(self._f_source)

        if self._dump:
            if self._no_move:
                self._d_plot_ij.append(np.zeros(100))
            else:
                self._d_plot_ij.append(f_i_j_given_d[:100])
            p_i = np.zeros(100)
            p_v = np.zeros(100)
            p_post = np.zeros(100)
            for i in range(100):
                if 0 < self._cur_pos - 50 + i < self._sax_length:
                    p_i[i] = f_i_given_d[self._cur_pos - 50 + i]
                    p_v[i] = f_v_given_i[self._cur_pos - 50 + i]
                    p_post[i] = self._f_source[self._cur_pos - 50 + i]
            self._d_plot_i.append(p_i)
            self._d_plot_v.append(p_v)
            self._d_plot_post.append(p_post)

        # determine no_move flag
        # if no sound in performance, do not push forward before the start of a note
        if self._cur_pos < self._sax_length - 1:
            # self._no_move = a_pitch == signal_processing.PitchProcessorCore.NO_PITCH and \
            #                 self._sax_pitch[self._cur_pos + 1] != signal_processing.PitchProcessorCore.NO_PITCH
            if a_pitch == signal_processing.PitchProcessorCore.NO_PITCH and \
                    self._sax_pitch[self._cur_pos + 1] != signal_processing.PitchProcessorCore.NO_PITCH:
                self._no_move = True
            elif a_pitch == signal_processing.PitchProcessorCore.NO_PITCH and \
                    self._sax_pitch[self._cur_pos] != signal_processing.PitchProcessorCore.NO_PITCH:
                self._no_move = True
            else:
                self._no_move = False

        # actively close audio input if follows to the end
        # if self._cur_pos >= self._sax_length - 1:
        if self._cur_pos >= 6000:
            a_input.kill()
            # kill auto accompaniment process
            self._auto_acco.enqueue((-1, -1, -1))

        # periodically housekeeping tasks
        # report information to downstream module
        if (self._cur_pos - self._prev_report_pos) / 100 > 60 / self._estimated_tempo * 1:
            # (timestamp, position in beat, confidence)
            self._auto_acco.enqueue((
                a_time,
                self._sax_time[self._cur_pos] * self._score_tempo / 60,  # score_tempo is in BPM, we need BPS
                self._f_source[self._cur_pos]
            ))
            self._prev_report_pos = self._cur_pos
        # re-estimate tempo
        if (self._cur_pos - self._prev_tempo_pos) / 100 > 60 / self._estimated_tempo * 2:
            estimated_tempo = ScoreFollower._estimate_tempo(score_tempo=self._score_tempo,
                                                            delta_pos=self._cur_pos - self._prev_tempo_pos,
                                                            delta_time=a_time - self._prev_tempo_time)
            estimated_tempo = min(estimated_tempo, self._tempo_ub)
            estimated_tempo = max(estimated_tempo, self._tempo_lb)
            self._estimated_tempo = estimated_tempo
            self._prev_tempo_pos = self._cur_pos
            self._prev_tempo_time = a_time
            print('SF: tempo set to', estimated_tempo)

        if self._dump:
            while self._d_midi_pos < len(self._d_midi_ref.instruments[0].notes) and \
                    self._sax_time[self._cur_pos] >= self._d_midi_ref.instruments[0].notes[self._d_midi_pos].start:
                o_note = self._d_midi_ref.instruments[0].notes[self._d_midi_pos]
                start = a_time - a_input.start_time
                self._d_midi_new.instruments[0].notes.append(pretty_midi.Note(velocity=o_note.velocity,
                                                                              pitch=o_note.pitch,
                                                                              start=start,
                                                                              end=start + o_note.end - o_note.start))
                self._d_midi_pos += 1
            self._d_perf.append(time.perf_counter() - self._perf_start)

    @staticmethod
    def _compute_f_i_j_given_d(time_axis, d, score_tempo, estimated_tempo):
        rate_ratio = estimated_tempo / score_tempo if estimated_tempo > 0 else score_tempo / 1e-5
        sigma_square = math.log(1 / (10 * d) + 1)
        sigma = math.sqrt(sigma_square)
        a = np.true_divide(1, np.multiply(time_axis, sigma * math.sqrt(2 * math.pi)), where=time_axis != 0)
        b = np.add(np.log(time_axis, where=time_axis != 0), 0.5 * sigma_square - math.log(rate_ratio * d))
        b = np.exp(np.true_divide(-np.square(b), 2 * sigma_square))
        f_i_j_given_d = np.multiply(a, b)
        # remove the possible np.nan element in the beginning, otherwise normalization will fail
        f_i_j_given_d[time_axis == 0] = 0
        f_i_j_given_d = normalize(f_i_j_given_d)
        return f_i_j_given_d

    @staticmethod
    def _compute_f_i_given_d(f_source, f_i_j_given_d, cur_pos, axis_length):
        # avoid overflow
        left = max(0, cur_pos - 1000)
        right = min(cur_pos + 1000, axis_length)
        f_i_given_d = np.zeros(axis_length)
        f_source_w = f_source[left:right]
        f_i_j_given_d_w = f_i_j_given_d[:right - left]
        f_i_given_d_w = np.convolve(f_source_w, f_i_j_given_d_w)
        f_i_given_d_w = f_i_given_d_w[:right - left]  # slice to window size
        f_i_given_d[left:right] = f_i_given_d_w
        f_i_given_d = normalize(f_i_given_d)
        return f_i_given_d

    # TODO: rewrite this to better model
    @staticmethod
    def _compute_f_v_given_i(pitch_axis, onset_axis, cur_pos, axis_length, audio_pitch, audio_onset, pitch_proc, w):
        f_v_given_i = np.zeros(axis_length)
        left = max(0, cur_pos - 200)
        right = min(cur_pos + 200, axis_length)
        for i in range(left, right):
            if audio_pitch == signal_processing.PitchProcessorCore.NO_PITCH:
                # performance side makes no sound
                if pitch_axis[i] == signal_processing.PitchProcessorCore.NO_PITCH:
                    # score side also makes no sound
                    f_v_given_i[i] = 0.1
                else:
                    # score side makes sound
                    f_v_given_i[i] = 1e-11
            else:
                # performance side makes sound
                if pitch_axis[i] == signal_processing.PitchProcessorCore.NO_PITCH:
                    # score side makes no sound
                    f_v_given_i[i] = 1e-11
                else:
                    # score side also makes sound
                    f_v_given_i[i] = math.pow(
                        math.pow(norm_pdf(pitch_proc.result(pitch_axis[i]), pitch_axis[i], 1), w[0])
                        * math.pow(similarity(audio_onset, onset_axis[i]), w[1]),
                        w[2]
                    )
        return f_v_given_i

    @staticmethod
    def _estimate_tempo(score_tempo, delta_pos, delta_time):
        return delta_pos * score_tempo * ScoreFollower.RESOLUTION / delta_time


class AutoAccompaniment:
    # fax_ means score Following AXis
    DEPTH = 5
    LATENCY = 0.02

    def __init__(self, config, score_bpm):
        self._config = config
        self._score_bps = score_bpm / 60
        self._fax_time = np.array([float(-x) for x in range(AutoAccompaniment.DEPTH)])
        self._fax_pos = np.array([-self._score_bps * x for x in range(AutoAccompaniment.DEPTH)])
        self._fax_conf = np.full(AutoAccompaniment.DEPTH, 0.001)
        self._first_run = True
        self._start_time = 0

        self._worker = None
        self._msg_queue = multiprocessing.Queue()

    def _worker_func(self):
        # run in a separate process
        if self._config['acco_mode'] == 0:
            self._player = audio_io.MIDIPlayer(self._config)
        self._player.connect_to_proc(self._proc)
        self._player.loop()  # blocking

    def _proc(self, a_time, a_output: audio_io.AudioOutput):
        # called by audio output, in separate process
        # MEMORY IS NOT SHARED! ONLY _msg_queue IS SAFE! USE OTHER VARIABLES VERY CAREFULLY!
        if self._first_run:
            self._first_run = False
            self._start_time = a_time

        has_update = False
        while self._msg_queue.qsize() > 0:
            has_update = True
            # (timestamp, position in beat, confidence)
            frame = self._msg_queue.get()
            # kill signal from score following
            if frame == (-1, -1, -1):
                self._player.kill()
            self._fax_time = np.roll(self._fax_time, 1)
            self._fax_time[0] = frame[0] - self._start_time
            self._fax_pos = np.roll(self._fax_pos, 1)
            self._fax_pos[0] = frame[1]
            self._fax_conf = np.roll(self._fax_conf, 1)
            self._fax_conf[0] = frame[2]
        if has_update:
            # all beats mean beats in performance score
            wls_model = sm.WLS(self._fax_pos, sm.add_constant(self._fax_time), weights=self._fax_conf)
            perf_tempo = wls_model.fit().params[1]  # estimated performance tempo in BPS, use weighted linear regression
            # plan to converge after 4 beats, beat / (beat/second) = second
            target_pos = (a_time - self._start_time - self._fax_time[0]) * perf_tempo + 4 + self._fax_pos[0]
            follow_time = 4 / perf_tempo - AutoAccompaniment.LATENCY
            follow_tempo = (target_pos - a_output.current_position) / follow_time  # new tempo in BPS
            follow_tempo = max(0, follow_tempo)
            follow_tempo_ratio = follow_tempo / self._score_bps  # ratio compared to original performance tempo
            a_output.change_tempo_ratio(follow_tempo_ratio)
            print('\t\t\t\tAA: perf_tempo', perf_tempo, 'current_pos', a_output.current_position, 'target_pos',
                  target_pos, 'follow_time', follow_time,
                  'follow_tempo', follow_tempo)
            print(self._fax_time)
            print(self._fax_pos)
            print(self._fax_conf)

    def enqueue(self, item):
        # called by score following processor, in separate thread
        # (timestamp, position in beat, confidence)
        self._msg_queue.put(item)

    def loop(self):
        # called by main thread
        # creating a new process works similar to fork() in Linux/Unix, therefore, assign values before this action
        self._worker = multiprocessing.Process(target=self._worker_func, args=(), daemon=False)
        self._worker.start()


if __name__ == '__main__':
    test_config = {
        'name': 'debug_3_fix_2',
        'perf_mode': 0,
        'perf_audio': 'audio/audio3.wav',
        'perf_sr': 44100,
        'perf_chunk': 1024,
        'score_midi': 'midi/midi3.mid',
        'acco_mode': 0,
        'acco_midi': 'midi/midi3.mid',
        'acco_audio': 'audio/audio3.wav',
        'dump_mic': False,
        'dump_sf': True
    }
    test_sf = ScoreFollower(test_config)
    test_sf.loop()
