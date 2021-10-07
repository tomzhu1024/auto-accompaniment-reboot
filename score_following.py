import json
import math
import shutil
import signal
import sys
import time

import numpy as np
import pretty_midi
from rich.live import Live
from rich.table import Table

import global_config
from utils import audio_io, shared_utils, udp_pipe, signal_processing

IGNORE_OBSERVATION = False
MOCK_RATE_RATIO = None


def generate_run_info_table(params=None) -> Table:
    table = Table()
    table.add_column('Audio Pitch')
    table.add_column('Audio Onset')
    table.add_column('Real Time /s')
    table.add_column('Scr. Time /s')
    table.add_column('Scr. Len. /s')
    table.add_column('Est. Tempo /bpm')
    table.add_column('Scr. Tempo /bpm')
    table.add_column('NO_MOVE')
    if params is None:
        table.add_row('---', '---', '---', '---', '---', '---', '---', '---')
    else:
        audio_pitch, audio_onset, real_time, score_time, score_length, estimated_tempo, score_tempo, no_move = params
        if estimated_tempo > score_tempo:
            estimated_tempo_color = '[blue]'
        elif estimated_tempo < score_tempo:
            estimated_tempo_color = '[yellow]'
        else:
            estimated_tempo_color = '[white]'
        table.add_row(f"{audio_pitch:.2f}",
                      '[green]True' if audio_onset else '[red]False',
                      f"{real_time:.3f}",
                      f"{score_time:.3f}",
                      f"{score_length:.3f}",
                      f"{estimated_tempo_color}{estimated_tempo:.4f}",
                      f"{score_tempo:.4f}",
                      '[red]True' if no_move else '[green]False')
    return table


class ScoreFollower:
    def __init__(self, config):
        self._config = config
        # the duration of each audio chunk
        self._audio_interval = self._config['perf_chunk'] / self._config['perf_sr']
        # the time distance between two adjacent data points on the density functions
        self._resolution = self._audio_interval / self._config['resolution_multiple']
        # the number of data points the density functions contains within one second
        self._points_per_second = math.ceil(1 / self._resolution)
        # `sax_` means `Score AXis`
        self._score_tempo, self._sax_time, self._sax_pitch, self._sax_onset, self._sax_length, \
        self._tempo_estimation_pos_lst = self._load_score(self._config['score_midi'], self._resolution)
        self._f_source = np.zeros(self._sax_length)
        self._f_source[0] = 1
        self._cur_pos = 0
        self._estimated_tempo = self._score_tempo
        self._prev_tempo_pos = 0
        self._prev_tempo_time = 0
        self._prev_report_time = 0
        self._first_run = True
        self._no_move = False
        # use this x-axis to compute expectation
        self._f_x_axis = np.arange(self._sax_length)
        # record the index of the list of tempo estimation position
        self._tempo_estimation_pos_idx = 0

        self._tempo_ub = 1.3 * self._score_tempo
        self._tempo_lb = 0.7 * self._score_tempo

        if self._config['perf_mode'] == 0:
            self._audio_input = audio_io.WaveFileInput(self._config)
        elif self._config['perf_mode'] == 1:
            self._audio_input = audio_io.MicrophoneInput(self._config)
        self._audio_input.connect_to_proc(self._proc)
        self._pitch_proc = signal_processing.PitchProcessor(self._config)
        self._onset_proc = signal_processing.OnsetProcessor(self._config)
        self._msg_sender = udp_pipe.UDPSender()
        self._live_display = None

        def signal_handler(_signal, _frame):
            # notify peer process and gracefully shutdown IPC channel
            self._msg_sender({
                'type': 'stop'
            })
            self._msg_sender.close()
            sys.exit(-1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # dump information for debug purpose
        self._dump = self._config['dump']
        if self._dump:
            # dump score following output as MIDI
            self._dmp_midi_origin = pretty_midi.PrettyMIDI(self._config['score_midi'])
            self._dmp_midi_output = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))
            self._dmp_midi_output.instruments.append(piano)
            self._dmp_midi_pos = 0
            # dump detected audio features
            self._dmp_audio_pitch = []
            self._dmp_audio_onset = []
            # dump density functions
            self._dmp_pdf_ij = []
            self._dmp_pdf_i = []
            self._dmp_pdf_v = []
            self._dmp_pdf_post = []
            # dump calculation results
            self._dmp_real_time = []
            self._dmp_cur_pos = []
            self._dmp_confidence = []
            # dump periodical housekeeping works
            self._dmp_report_pos = []
            self._dmp_estimate_tempo = []
            # dump time points during execution
            self._dmp_exec_time = []

    def _load_score(self, midi_path, resolution):
        midi_file = pretty_midi.PrettyMIDI(midi_path)
        instrument = midi_file.instruments[0]
        score_tempo = shared_utils.average(midi_file.get_tempo_changes()[1])  # BPM

        sax_length = math.ceil(max([note.end for note in instrument.notes]) / resolution) + 1  # include 0
        sax_time = np.arange(0, sax_length * resolution, resolution)
        sax_pitch = np.zeros(sax_length)
        sax_onset = np.zeros(sax_length)
        pitches = np.full(sax_length, signal_processing.PitchProcessorCore.NO_PITCH)
        for note in instrument.notes:
            start = math.ceil(note.start / resolution)
            end = math.ceil(note.end / resolution) + 1  # end will never go out of range
            # truncate of note is too long
            if end - start > int(1 / resolution):
                end = start + int(1 / resolution)
            pitches[start: end] = note.pitch
            sax_onset[start] = 1
        score_pitch_proc = signal_processing.PitchProcessorCore()
        for i in range(sax_length):
            sax_pitch[i] = score_pitch_proc(pitches[i])

        # generate list of time-on-score for tempo estimation
        tempo_est_pos_lst = np.arange(60 / score_tempo, sax_length * resolution, 60 / score_tempo)
        # convert time to index
        tempo_est_pos_lst = [round(x / resolution) for x in tempo_est_pos_lst]

        return score_tempo, sax_time, sax_pitch, sax_onset, sax_length, tempo_est_pos_lst

    def loop(self):
        with Live(generate_run_info_table(), refresh_per_second=4, transient=True) as live:
            # share `live` within the class instance
            self._live_display = live
            if self._dump:
                cur_time = time.time()
                self._dmp_exec_time.append(cur_time)
                self._dmp_exec_time.append(cur_time)
            self._audio_input.loop()  # blocking, block main process for reading, use separate thread for processing
        # some cleaning work after loop
        # kill auto accompaniment process
        self._msg_sender({
            'type': 'stop'
        })
        self._msg_sender.close()
        # save dump data to files
        if self._dump:
            shared_utils.check_dir('output', self._config['name'])

            # save configurations to JSON file
            with open(f"output/{self._config['name']}/global_config.json", 'w') as fs:
                fs.write(json.dumps(self._config))

            # save audio input
            if self._config['perf_mode'] == 0:
                # input source is WAV file, just copy the file
                shutil.copyfile(self._config['perf_audio'], f"output/{self._config['name']}/audio_input.wav")
            elif self._config['perf_mode'] == 1:
                # input source is microphone, save the buffer to file
                # note that only `MicrophoneInput` supports this method
                self._audio_input.save_to_file(f"output/{self._config['name']}/audio_input.wav")

            # write score following result to MIDI file
            self._dmp_midi_output.write(f"output/{self._config['name']}/sf_output.mid")

            # copy original performance MIDI file
            shutil.copyfile(self._config['score_midi'], f"output/{self._config['name']}/sf_origin.mid")

            # dump data points about calculation
            np.save(f"output/{self._config['name']}/sf_audio_pitch.npy", self._dmp_audio_pitch)
            np.save(f"output/{self._config['name']}/sf_audio_onset.npy", self._dmp_audio_onset)
            np.save(f"output/{self._config['name']}/sf_pdf_ij.npy", self._dmp_pdf_ij)
            np.save(f"output/{self._config['name']}/sf_pdf_i.npy", self._dmp_pdf_i)
            np.save(f"output/{self._config['name']}/sf_pdf_v.npy", self._dmp_pdf_v)
            np.save(f"output/{self._config['name']}/sf_pdf_post.npy", self._dmp_pdf_post)
            np.save(f"output/{self._config['name']}/sf_real_time.npy", self._dmp_real_time)
            np.save(f"output/{self._config['name']}/sf_cur_pos.npy", self._dmp_cur_pos)
            np.save(f"output/{self._config['name']}/sf_confidence.npy", self._dmp_confidence)
            np.save(f"output/{self._config['name']}/sf_report_pos.npy", self._dmp_report_pos)
            np.save(f"output/{self._config['name']}/sf_estimate_tempo.npy", self._dmp_estimate_tempo)
            np.save(f"output/{self._config['name']}/sf_exec_time.npy", self._dmp_exec_time)

            # dump data points about score
            np.save(f"output/{self._config['name']}/sf_sax_time.npy", self._sax_time)
            np.save(f"output/{self._config['name']}/sf_sax_pitch.npy", self._sax_pitch)
            np.save(f"output/{self._config['name']}/sf_sax_onset.npy", self._sax_onset)

    def _proc(self, a_time, prev_a_time, a_data, a_input: audio_io.AudioInput):
        if self._dump:
            self._dmp_exec_time.append(time.time())

        a_relative_time = a_time - a_input.start_time
        # `a_` means `Audio`
        # called by audio input, execute in thread pool
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
        # `f_source` and `cur_pos` remains unchanged here
        if self._f_source[self._cur_pos] > 0.1:
            w = (0.95, 0.05, 0.5)
        else:
            w = (0.7, 0.3, 0.3)

        # prior
        if self._no_move:
            f_i_j_given_d = None
            f_i_given_d = self._f_source
        else:
            f_i_j_given_d = self._compute_f_i_j_given_d(time_axis=self._sax_time,
                                                        d=a_time - prev_a_time,
                                                        score_tempo=self._score_tempo,
                                                        estimated_tempo=self._estimated_tempo)
            f_i_given_d = self._compute_f_i_given_d(f_source=self._f_source,
                                                    f_i_j_given_d=f_i_j_given_d,
                                                    cur_pos=self._cur_pos,
                                                    axis_length=self._sax_length)
            f_i_given_d = self._normalize(f_i_given_d)
        # update position
        self._cur_pos = round(self._f_x_axis.dot(f_i_given_d))

        # observation
        f_v_given_i = self._compute_f_v_given_i(pitch_axis=self._sax_pitch,
                                                onset_axis=self._sax_onset,
                                                cur_pos=self._cur_pos,
                                                axis_length=self._sax_length,
                                                audio_pitch=a_pitch,
                                                audio_onset=a_onset,
                                                pitch_proc=self._pitch_proc,
                                                w=w)
        # posterior
        if not IGNORE_OBSERVATION:
            self._f_source = f_i_given_d * f_v_given_i
        else:
            self._f_source = f_i_given_d
        self._f_source = self._gate_mask(self._f_source, center=self._cur_pos,
                                         half_size=math.ceil(self._config['gate_post'] / 2 * self._points_per_second))
        self._f_source = self._normalize(self._f_source)
        # update position again
        self._cur_pos = round(self._f_x_axis.dot(self._f_source))

        # forcefully close audio input if it follows to the end
        if self._cur_pos >= self._sax_length - 1:
            # this branch won't be executed if the audio input closes on its own
            a_input.kill()

        # periodically housekeeping tasks
        #
        # report information to accompaniment module
        if a_time - self._prev_report_time > self._config['pos_report_interval']:
            # use position in beat, therefore the tempo variance will not affect the position
            self._msg_sender({
                'type': 'update',
                'time': a_time,
                'pos': self._sax_time[self._cur_pos] / 60 * self._score_tempo,  # sec / 60 * BPM = beat
                'conf': self._f_source[self._cur_pos]
            })
            self._prev_report_time = a_time
            # update dump data, use `1` to mark reporting event
            if self._dump:
                self._dmp_report_pos.append(1)
        elif self._dump:
            # update dump data, use `0` to mark no reporting event
            self._dmp_report_pos.append(0)
        # re-estimate tempo
        if self._cur_pos >= self._tempo_estimation_pos_lst[self._tempo_estimation_pos_idx]:
            self._estimated_tempo = self._estimate_tempo(score_tempo=self._score_tempo,
                                                         delta_pos=self._cur_pos - self._prev_tempo_pos,
                                                         delta_time=a_time - self._prev_tempo_time)
            self._estimated_tempo = min(self._estimated_tempo, self._tempo_ub)
            self._estimated_tempo = max(self._estimated_tempo, self._tempo_lb)
            self._prev_tempo_pos = self._cur_pos
            self._prev_tempo_time = a_time
            # update `_tempo_estimation_pos_idx`
            while self._cur_pos >= self._tempo_estimation_pos_lst[self._tempo_estimation_pos_idx]:
                self._tempo_estimation_pos_idx += 1
            # update dump data, record estimated tempo
            if self._dump:
                self._dmp_estimate_tempo.append(self._estimated_tempo)
        elif self._dump:
            # update dump data, use `-1` to mark no estimation event
            self._dmp_estimate_tempo.append(-1)

        # update dump data
        if self._dump:
            # update output MIDI
            while self._dmp_midi_pos < len(self._dmp_midi_origin.instruments[0].notes) and \
                    self._sax_time[self._cur_pos] >= \
                    self._dmp_midi_origin.instruments[0].notes[self._dmp_midi_pos].start:
                o_note = self._dmp_midi_origin.instruments[0].notes[self._dmp_midi_pos]
                start = a_relative_time
                self._dmp_midi_output.instruments[0].notes.append(pretty_midi.Note(
                    velocity=o_note.velocity,
                    pitch=o_note.pitch,
                    start=start,
                    end=start + o_note.end - o_note.start))
                self._dmp_midi_pos += 1
            # update audio features
            self._dmp_audio_pitch.append(a_pitch)
            self._dmp_audio_onset.append(int(a_onset))
            # update density functions
            #
            # the window size of the density function `f_IJ_given_D` is determined by the configuration field
            # `window_ij`
            # let the window size of the density function `f_I_given_D` and posterior same as `f_IJ_given_D`
            window_size_ij_i_post = math.ceil(self._config['window_ij'] / 2 * self._points_per_second) * 2
            # the window size of the density function `f_V_given_D` is determined by the configuration field `window_v`
            window_size_v = math.ceil(self._config['window_v'] / 2 * self._points_per_second) * 2
            if self._no_move:
                # if `no_move` is activated, `f_IJ_given_D` is meaningless, fill `-1` here
                self._dmp_pdf_ij.append(np.full(window_size_ij_i_post, -1))
            else:
                # `f_IJ_given_D` starts from `0`, so it is enough to truncate the tail
                self._dmp_pdf_ij.append(f_i_j_given_d[:window_size_ij_i_post])
            p_i = np.zeros(window_size_ij_i_post)
            p_v = np.zeros(window_size_v)
            p_post = np.zeros(window_size_ij_i_post)
            for i in range(window_size_ij_i_post):
                pos = self._cur_pos - window_size_ij_i_post // 2 + i
                if 0 < pos < self._sax_length:
                    p_i[i] = f_i_given_d[pos]
                    p_post[i] = self._f_source[pos]
            for i in range(window_size_v):
                pos = self._cur_pos - window_size_v // 2 + i
                if 0 < pos < self._sax_length:
                    p_v[i] = f_v_given_i[pos]
            self._dmp_pdf_i.append(p_i)
            self._dmp_pdf_v.append(p_v)
            self._dmp_pdf_post.append(p_post)
            # update calculation results
            self._dmp_real_time.append(a_time)
            self._dmp_cur_pos.append(self._cur_pos)
            self._dmp_confidence.append(self._f_source[self._cur_pos])
            # forcefully close audio input if it reaches truncating time
            if 0 < self._config['trunc_time'] <= a_relative_time:
                # the audio input kills itself by setting a flag to `False` internally, so killing twice is still safe
                a_input.kill()

        # determine no_move flag
        # if no sound in performance, do not push forward before the start of a note
        if not IGNORE_OBSERVATION and self._cur_pos < self._sax_length - 1:
            self._no_move = a_pitch == signal_processing.PitchProcessorCore.NO_PITCH and (
                    self._sax_pitch[self._cur_pos + 1] != signal_processing.PitchProcessorCore.NO_PITCH or
                    self._sax_pitch[self._cur_pos] != signal_processing.PitchProcessorCore.NO_PITCH)

        # update live display
        self._live_display.update(generate_run_info_table((a_pitch,
                                                           a_onset,
                                                           a_relative_time,
                                                           self._sax_time[self._cur_pos],
                                                           self._sax_time[-1],
                                                           self._estimated_tempo,
                                                           self._score_tempo,
                                                           self._no_move)))

        if self._dump:
            self._dmp_exec_time.append(time.time())

    def _compute_f_i_j_given_d(self, time_axis, d, score_tempo, estimated_tempo):
        rate_ratio = estimated_tempo / score_tempo if estimated_tempo > 0 else 1e-5 / score_tempo
        sigma_square = math.log(1 / (100 * d) + 1)
        sigma = math.sqrt(sigma_square)
        mu = math.log(d * rate_ratio) - sigma_square / 2
        f_i_j_given_d = np.divide(1, time_axis, where=time_axis != 0) * sigma * math.sqrt(2 * math.pi) * np.exp(
            - ((np.log(time_axis, where=time_axis != 0) - mu) ** 2 / (2 * sigma ** 2)))
        f_i_j_given_d[0] = 0
        f_sum = np.sum(f_i_j_given_d)
        if f_sum != 0:
            f_i_j_given_d = np.divide(f_i_j_given_d, f_sum)
        return f_i_j_given_d

    def _compute_f_i_given_d(self, f_source, f_i_j_given_d, cur_pos, axis_length):
        # apply a window here to enhance performance, but remember to avoid index overflow
        half_win_size = math.ceil(self._config['window_ij'] / 2 * self._points_per_second)
        left = max(0, cur_pos - half_win_size)
        right = min(cur_pos + half_win_size, axis_length)
        f_i_given_d = np.zeros(axis_length)
        f_source_w = f_source[left:right]
        f_i_j_given_d_w = f_i_j_given_d[:right - left]
        f_i_given_d_w = np.convolve(f_source_w, f_i_j_given_d_w)
        f_i_given_d_w = f_i_given_d_w[:right - left]  # slice to window size
        f_i_given_d[left:right] = f_i_given_d_w
        f_sum = np.sum(f_i_given_d)
        if f_sum != 0:
            f_i_given_d = np.divide(f_i_given_d, f_sum)
        return f_i_given_d

    def _compute_f_v_given_i(self, pitch_axis, onset_axis, cur_pos, axis_length, audio_pitch, audio_onset, pitch_proc,
                             w):
        f_v_given_i = np.zeros(axis_length)
        # apply a window here to enhance performance, but remember to avoid index overflow
        half_win_size = math.ceil(self._config['window_v'] / 2 * self._points_per_second)
        left = max(0, cur_pos - half_win_size)
        right = min(cur_pos + half_win_size, axis_length)
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
                        math.pow(self._norm_pdf(pitch_proc.result(pitch_axis[i]), pitch_axis[i], 1), w[0])
                        * math.pow(self._similarity(audio_onset, onset_axis[i]), w[1]),
                        w[2]
                    )
        return f_v_given_i

    def _estimate_tempo(self, score_tempo, delta_pos, delta_time):
        return delta_pos * score_tempo * self._resolution / delta_time

    def _normalize(self, array):
        return np.true_divide(array, np.sum(array))

    def _norm_pdf(self, x, mean, sd=1):
        var = sd ** 2
        denom = (2 * math.pi * var) ** 0.5
        num = math.exp(-(x - mean) ** 2 / (2 * var))
        return num / denom

    def _similarity(self, left, right):
        return (min(left, right) + 1e-6) / (max(left, right) + 1e-6)

    def _gate_mask(self, array, center, half_size):
        left_index = max(0, center - half_size)
        right_index = min(center + half_size, len(array))
        array[:left_index] = 0
        array[right_index:] = 0
        return array


if __name__ == '__main__':
    app = ScoreFollower(global_config.config)
    app.loop()
