import math
import time

import numpy as np
import pretty_midi

import audio_io
import global_config
import shared_utils
import signal_processing
import udp_pipe


class ScoreFollower:
    # sax means Score AXis
    # a means Audio
    SUPER_SAMPLING_FACTOR = 4
    RESOLUTION = 2048 / 44100 / SUPER_SAMPLING_FACTOR

    def __init__(self, config):
        self._midi_path = config['score_midi']
        self._score_tempo, self._sax_time, self._sax_pitch, self._sax_onset, self._sax_length = \
            ScoreFollower._load_score(self._midi_path, ScoreFollower.RESOLUTION)
        self._f_source = np.zeros(self._sax_length)
        self._f_source[0] = 1.0
        self._cur_pos = 0
        self._estimated_tempo = self._score_tempo
        # self._prev_report_pos = 0
        self._prev_report_time = 0
        self._prev_tempo_pos = 0
        self._prev_tempo_time = 0
        self._first_run = True
        self._no_move = False
        self._f_x_axis = np.arange(self._sax_length)

        self._tempo_ub = 1.3 * self._score_tempo
        self._tempo_lb = 0.7 * self._score_tempo

        if config['perf_mode'] == 0:
            self._audio_input = audio_io.WaveFileInput(config)
        elif config['perf_mode'] == 1:
            self._audio_input = audio_io.MicrophoneInput(config)
        self._audio_interval = config['perf_chunk'] / config['perf_sr']
        self._audio_input.connect_to_proc(self._proc)
        self._pitch_proc = signal_processing.PitchProcessor(config)
        self._onset_proc = signal_processing.OnsetProcessor(config)
        self._msg_sender = udp_pipe.UDPSender()

        # dump information for debug purpose
        self._dump = config['dump']
        if self._dump:
            self._dump_dir = config['name']
            # dump midi
            self._d_midi_ref = pretty_midi.PrettyMIDI(self._midi_path)
            self._d_midi_new = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))
            self._d_midi_new.instruments.append(piano)
            self._d_midi_pos = 0
            # dump time-position graph
            self._d_progress_time = []
            self._d_progress_pos = []
            self._d_progress_conf = []
            self._d_progress_report = []
            # dump snapshots
            self._d_snapshot_ij = []
            self._d_snapshot_i = []
            self._d_snapshot_v = []
            self._d_snapshot_posterior = []
            # dump time cost
            self._d_time_cost = []
            self._perf_start = 0
            # truncating time
            self._trunc_time = config['trunc_time']

        # DEBUG PRINT
        print("Score Following Module initialized. Profile Name: %s. Audio Interval: %.4f." % (
            config['name'], self._audio_interval
        ))

    @staticmethod
    def _load_score(midi_path, resolution):
        midi_file = pretty_midi.PrettyMIDI(midi_path)
        instrument = midi_file.instruments[0]
        score_tempo = shared_utils.average(midi_file.get_tempo_changes()[1])

        sax_length = int(math.ceil(max([note.end for note in instrument.notes]) / resolution)) + 1  # include 0
        sax_time = np.arange(0, sax_length * resolution, resolution)
        sax_pitch = np.zeros(sax_length)
        sax_onset = np.zeros(sax_length)
        pitches = np.full(sax_length, signal_processing.PitchProcessorCore.NO_PITCH)
        for note in instrument.notes:
            start = int(math.ceil(note.start / resolution))
            end = int(math.ceil(note.end / resolution)) + 1  # end will never go out of range
            # truncate of note is too long
            if end - start > int(1 / resolution):
                end = start + int(1 / resolution)
            pitches[start: end] = note.pitch
            sax_onset[start] = 1
        score_pitch_proc = signal_processing.PitchProcessorCore()
        for i in range(sax_length):
            sax_pitch[i] = score_pitch_proc(pitches[i])
        return score_tempo, sax_time, sax_pitch, sax_onset, sax_length

    def loop(self):
        self._audio_input.loop()  # blocking, block main process for reading, use separate thread for processing
        # some cleaning work after loop
        # kill auto accompaniment process
        self._msg_sender({
            'type': 'stop'
        })
        self._msg_sender.close()
        if self._dump:
            shared_utils.check_dir('output', self._dump_dir)
            self._d_midi_new.write('output/%s/sf_result.mid' % self._dump_dir)
            np.save('output/%s/sf_snapshot_ij.npy' % self._dump_dir, self._d_snapshot_ij)
            np.save('output/%s/sf_snapshot_i.npy' % self._dump_dir, self._d_snapshot_i)
            np.save('output/%s/sf_snapshot_v.npy' % self._dump_dir, self._d_snapshot_v)
            np.save('output/%s/sf_snapshot_posterior.npy' % self._dump_dir, self._d_snapshot_posterior)
            np.save('output/%s/sf_time_cost.npy' % self._dump_dir, self._d_time_cost)
            np.save('output/%s/sf_progress_time.npy' % self._dump_dir, self._d_progress_time)
            np.save('output/%s/sf_progress_pos.npy' % self._dump_dir, self._d_progress_pos)
            np.save('output/%s/sf_progress_conf.npy' % self._dump_dir, self._d_progress_conf)
            np.save('output/%s/sf_progress_report.npy' % self._dump_dir, self._d_progress_report)

    def _proc(self, a_time, prev_a_time, a_data, a_input: audio_io.AudioInput):
        # called by audio input, execute in thread pool
        if self._dump:
            self._perf_start = time.perf_counter()

        if self._first_run:
            self._first_run = False
            self._prev_report_time = a_input.start_time
            self._prev_tempo_time = a_input.start_time
            # send message to activate auto accompaniment
            self._msg_sender({
                'type': 'start',
                'time': a_time,
                'tempo': self._score_tempo
            })

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
            f_i_given_d = ScoreFollower._normalize(f_i_given_d)
        # update position
        # self._cur_pos = np.argmax(f_i_given_d)
        self._cur_pos = round(np.dot(self._f_x_axis, f_i_given_d))

        # observation
        f_v_given_i = ScoreFollower._compute_f_v_given_i(pitch_axis=self._sax_pitch,
                                                         onset_axis=self._sax_onset,
                                                         cur_pos=self._cur_pos,
                                                         axis_length=self._sax_length,
                                                         audio_pitch=a_pitch,
                                                         audio_onset=a_onset,
                                                         pitch_proc=self._pitch_proc,
                                                         w=w)
        f_v_given_i = ScoreFollower._normalize(f_v_given_i)
        # posterior
        self._f_source = f_i_given_d * f_v_given_i
        self._f_source = ScoreFollower._gate_mask(self._f_source, center=self._cur_pos,
                                                  half_size=50 * self.SUPER_SAMPLING_FACTOR)
        self._f_source = ScoreFollower._normalize(self._f_source)
        # update position again
        # self._cur_pos = np.argmax(self._f_source)
        self._cur_pos = round(np.dot(self._f_x_axis, self._f_source))

        if self._dump:
            # TODO: change dump size according to SUPER_SAMPLING_FACTOR
            if self._no_move:
                self._d_snapshot_ij.append(np.zeros(100))
            else:
                self._d_snapshot_ij.append(f_i_j_given_d[:100])
            p_i = np.zeros(100)
            p_v = np.zeros(100)
            p_post = np.zeros(100)
            for i in range(100):
                if 0 < self._cur_pos - 50 + i < self._sax_length:
                    p_i[i] = f_i_given_d[self._cur_pos - 50 + i]
                    p_v[i] = f_v_given_i[self._cur_pos - 50 + i]
                    p_post[i] = self._f_source[self._cur_pos - 50 + i]
            self._d_snapshot_i.append(p_i)
            self._d_snapshot_v.append(p_v)
            self._d_snapshot_posterior.append(p_post)
            self._d_progress_time.append(a_time)
            self._d_progress_pos.append(self._sax_time[self._cur_pos])
            self._d_progress_conf.append(self._f_source[self._cur_pos])
            # forcefully close audio input if it reaches truncating time
            if 0 < self._trunc_time <= a_time - a_input.start_time:
                # this branch won't be executed if the audio input closes on its own
                a_input.kill()

        # forcefully close audio input if it follows to the end
        if self._cur_pos >= self._sax_length - 1:
            # this branch won't be executed if the audio input closes on its own
            a_input.kill()

        # determine no_move flag
        # if no sound in performance, do not push forward before the start of a note
        if self._cur_pos < self._sax_length - 1:
            self._no_move = a_pitch == signal_processing.PitchProcessorCore.NO_PITCH and \
                            (self._sax_pitch[self._cur_pos + 1] != signal_processing.PitchProcessorCore.NO_PITCH or
                             self._sax_pitch[self._cur_pos] != signal_processing.PitchProcessorCore.NO_PITCH)

        # periodically housekeeping tasks
        # report information to accompaniment module
        # if (self._cur_pos - self._prev_report_pos) / 100 > 60 / self._estimated_tempo * 0.5:
        if a_time - self._prev_report_time > 1:
            # timestamp, position in beat, confidence
            self._msg_sender({
                'type': 'update',
                'time': a_time,
                'pos': self._sax_time[self._cur_pos] / 60 * self._score_tempo,  # sec / 60 * BPM = beat
                'conf': self._f_source[self._cur_pos]
            })
            # self._prev_report_pos = self._cur_pos
            self._prev_report_time = a_time
            if self._dump:
                self._d_progress_report.append(1)
        else:
            if self._dump:
                self._d_progress_report.append(0)
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
            self._d_time_cost.append(time.perf_counter() - self._perf_start)

            # DEBUG PRINT
            print('Pitch= {:>6}\tOnset= {:<3}\tPos= {:>8}\tBlocked= {:<3}\tCost= {:>7}\tRatio= {:>5}'.format(
                '%.3f' % a_pitch,
                '|==' if a_onset else '|',
                '%.3f' % self._sax_time[self._cur_pos],
                '|==' if self._no_move else '|',
                '%.2f%%' % (self._d_time_cost[-1] / self._audio_interval * 100),
                '%.3f' % (self._estimated_tempo / self._score_tempo)
            ))

    @staticmethod
    def _compute_f_i_j_given_d(time_axis, d, score_tempo, estimated_tempo):
        rate_ratio = score_tempo / estimated_tempo if estimated_tempo > 0 else score_tempo / 1e-5
        sigma_square = math.log(1 / (10 * d) + 1)
        sigma = math.sqrt(sigma_square)
        mu = math.log(d / rate_ratio) - sigma_square / 2
        f_i_j_given_d = np.true_divide(1, time_axis, where=time_axis != 0) * sigma * math.sqrt(2 * math.pi) * np.exp(
            - ((np.log(time_axis, where=time_axis != 0) - mu) ** 2 / (2 * sigma ** 2)))
        f_i_j_given_d[0] = 0
        return f_i_j_given_d

    @staticmethod
    def _compute_f_i_given_d(f_source, f_i_j_given_d, cur_pos, axis_length):
        # avoid overflow
        left = max(0, cur_pos - 1000 * ScoreFollower.SUPER_SAMPLING_FACTOR)
        right = min(cur_pos + 1000 * ScoreFollower.SUPER_SAMPLING_FACTOR, axis_length)
        f_i_given_d = np.zeros(axis_length)
        f_source_w = f_source[left:right]
        f_i_j_given_d_w = f_i_j_given_d[:right - left]
        f_i_given_d_w = np.convolve(f_source_w, f_i_j_given_d_w)
        f_i_given_d_w = f_i_given_d_w[:right - left]  # slice to window size
        f_i_given_d[left:right] = f_i_given_d_w
        return f_i_given_d

    @staticmethod
    def _compute_f_v_given_i(pitch_axis, onset_axis, cur_pos, axis_length, audio_pitch, audio_onset, pitch_proc, w):
        f_v_given_i = np.zeros(axis_length)
        left = max(0, cur_pos - 200 * ScoreFollower.SUPER_SAMPLING_FACTOR)
        right = min(cur_pos + 200 * ScoreFollower.SUPER_SAMPLING_FACTOR, axis_length)
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
                        math.pow(ScoreFollower._norm_pdf(pitch_proc.result(pitch_axis[i]), pitch_axis[i], 1), w[0])
                        * math.pow(ScoreFollower._similarity(audio_onset, onset_axis[i]), w[1]),
                        w[2]
                    )
        return f_v_given_i

    @staticmethod
    def _estimate_tempo(score_tempo, delta_pos, delta_time):
        return delta_pos * score_tempo * ScoreFollower.RESOLUTION / delta_time

    @staticmethod
    def _normalize(array):
        return np.true_divide(array, np.sum(array))

    @staticmethod
    def _norm_pdf(x, mean, sd=1):
        var = sd ** 2
        denom = (2 * math.pi * var) ** 0.5
        num = math.exp(-(x - mean) ** 2 / (2 * var))
        return num / denom

    @staticmethod
    def _similarity(left, right):
        return (min(left, right) + 1e-6) / (max(left, right) + 1e-6)

    @staticmethod
    def _gate_mask(array, center, half_size):
        left_index = max(0, center - half_size)
        right_index = min(center + half_size, len(array))
        array[:left_index] = 0
        array[right_index:] = 0
        return array


if __name__ == '__main__':
    app = ScoreFollower(global_config.config)
    app.loop()
