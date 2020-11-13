import time

import numpy as np
import pretty_midi
import statsmodels.api as sm

import audio_io
import shared_config
import shared_utils
import udp_pipe


class AutoAccompaniment:
    # fax means score Following AXis
    # a means Audio
    DEPTH = 5
    LATENCY = 0.02

    def __init__(self, config):
        self._midi_path = config['acco_midi']
        self._fax_time = np.array([float(-x) for x in range(AutoAccompaniment.DEPTH)])
        self._fax_pos = np.array([float(-x) for x in range(AutoAccompaniment.DEPTH)])
        self._fax_conf = np.full(AutoAccompaniment.DEPTH, 0.001)
        self._peer_start_time = 0
        self._peer_score_tempo = 0  # in BPS

        if config['acco_mode'] == 0:
            self._player = audio_io.MIDIPlayer(self._midi_path, audio_io.VirtualSynthesizer())
        elif config['acco_mode'] == 1:
            self._player = audio_io.MIDIPlayer(self._midi_path, audio_io.ExternalSynthesizer())
        else:
            raise Exception('unsupported accompaniment mode')
        self._player.connect_to_proc(self._proc)
        self._msg_receiver = udp_pipe.UDPReceiver()

        self._dump = config['dump']
        if self._dump:
            self._dump_dir = config['name']
            self._d_progress_time = []
            self._d_progress_pos = []
            self._d_progress_report = []

    def loop(self):
        print('Waiting for start signal...')
        # wait for start message
        while True:
            msg = self._msg_receiver()
            if msg is not None and msg['type'] == 'start':
                self._peer_start_time = msg['time']
                self._peer_score_tempo = msg['tempo'] / 60
                break
            time.sleep(audio_io.MIDIPlayer.TICK_INT)
        print('Playing...')
        self._player.loop()  # blocking
        print('Stopped')
        # some cleaning work
        self._msg_receiver.close()
        if self._dump:
            shared_utils.check_dir('output', self._dump_dir)
            # dump midi
            midi_ref = pretty_midi.PrettyMIDI(self._midi_path)
            midi_new = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))
            ax_time = [0] + [x - self._peer_start_time for x in self._d_progress_time]
            ax_pos = [0] + [x for x in self._d_progress_pos]
            going_notes = []
            for i in range(1, len(ax_pos)):
                for note in midi_ref.instruments[0].notes:
                    if ax_pos[i - 1] <= note.start <= ax_pos[i]:
                        going_notes.append((ax_time[i], note.pitch, note.velocity))
                    if ax_pos[i - 1] <= note.end <= ax_pos[i]:
                        for j in range(len(going_notes)):
                            if going_notes[j][1] == note.pitch and going_notes[j][2] == note.velocity:
                                piano.notes.append(pretty_midi.Note(start=going_notes[j][0],
                                                                    end=ax_time[i],
                                                                    pitch=going_notes[j][1],
                                                                    velocity=going_notes[j][2]))
                            del going_notes[j]
                            break
            midi_new.instruments.append(piano)
            midi_new.write('output/%s/ac_result.mid' % self._dump_dir)
            # dump progress
            np.save('output/%s/ac_progress_time.npy' % self._dump_dir, self._d_progress_time)
            np.save('output/%s/ac_progress_pos.npy' % self._dump_dir, self._d_progress_pos)
            np.save('output/%s/ac_progress_report.npy' % self._dump_dir, self._d_progress_report)

    def _proc(self, a_time, a_output):
        has_update = False
        while True:
            msg = self._msg_receiver()
            if msg is not None and msg['type'] == 'stop':
                a_output.kill()
                has_update = False
                break
            elif msg is not None and msg['type'] == 'update':
                self._fax_time = np.roll(self._fax_time, 1)
                self._fax_time[0] = msg['time'] - self._peer_start_time
                self._fax_pos = np.roll(self._fax_pos, 1)
                self._fax_pos[0] = msg['pos']
                self._fax_conf = np.roll(self._fax_conf, 1)
                self._fax_conf[0] = msg['conf']
                has_update = True
            else:
                break
        if has_update:
            # all beats mean beats in performance score
            wls_model = sm.WLS(self._fax_pos, sm.add_constant(self._fax_time), weights=self._fax_conf)
            fit_params = wls_model.fit().params
            perf_tempo = fit_params[1]  # estimated performance tempo in BPS, use weighted linear regression
            if perf_tempo > 0:
                current_pos = a_output.current_time * self._peer_score_tempo
                # the reason to use max() here is that UDP does not guarantee the order of arrival of packets
                follow_tempo = (fit_params[0] + (a_time - self._peer_start_time) * fit_params[1] + 4 - current_pos) / \
                               (4 / perf_tempo - AutoAccompaniment.LATENCY)
                follow_tempo = max(0, follow_tempo)  # in BPS
                # ratio compared to original performance tempo
                follow_tempo_ratio = follow_tempo / self._peer_score_tempo
                a_output.change_tempo_ratio(follow_tempo_ratio)
            else:
                a_output.change_tempo_ratio(0)
            if self._dump:
                self._d_progress_report.append(1)
        else:
            if self._dump:
                self._d_progress_report.append(0)
        if self._dump:
            self._d_progress_time.append(a_time)
            self._d_progress_pos.append(a_output.current_time)


if __name__ == '__main__':
    app = AutoAccompaniment(shared_config.config)
    app.loop()
