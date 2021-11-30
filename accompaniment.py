import shutil
import time

import numpy as np
import pretty_midi
import statsmodels.api as sm
from rich.console import Console
from rich.live import Live
from rich.table import Table

import global_config
from utils import audio_io, shared_utils, udp_pipe

console = Console()


def generate_run_info_table(params=None) -> Table:
    table = Table()
    table.add_column('Real Time /s')
    table.add_column('Scr. Time /s')
    table.add_column('Scr. Len. /s')
    table.add_column('Comp. Tempo /bps')
    table.add_column('Tempo Ratio')
    if params is None:
        table.add_row('---', '---', '---', '---', '---')
    else:
        real_time, score_time, score_length, computed_tempo, computed_tempo_ratio = params
        if computed_tempo_ratio == 0:
            computed_tempo_color = '[red]'
        elif computed_tempo_ratio > 1:
            computed_tempo_color = '[blue]'
        elif computed_tempo_ratio < 1:
            computed_tempo_color = '[yellow]'
        else:
            computed_tempo_color = '[white]'
        table.add_row(f"{real_time:.3f}", f"{score_time:.3f}", f"{score_length:.3f}",
                      f"{computed_tempo_color}{computed_tempo:.4f}",
                      f"{computed_tempo_color}{computed_tempo_ratio:.4f}")
    return table


class AutoAccompaniment:
    def __init__(self, config):
        self._config = config
        self._midi_path = self._config['acco_midi']
        # `fax_` means `score Following AXis`
        self._fax_time = np.array([float(-x) for x in range(self._config['regression_depth'])])
        self._fax_pos = np.array([float(-x) for x in range(self._config['regression_depth'])])
        self._fax_conf = np.full(self._config['regression_depth'], 0.001)
        self._peer_start_time = 0
        self._peer_score_tempo = 0  # in BPS
        self._perf_tempo = 0

        if self._config['acco_mode'] == 0:
            self._player = audio_io.MIDIPlayer(self._midi_path, audio_io.VirtualSynthesizer(r'resources/soundfont.sf2'))
        elif self._config['acco_mode'] == 1:
            self._player = audio_io.MIDIPlayer(self._midi_path, audio_io.ExternalSynthesizer(self._config))
        else:
            raise Exception('unsupported accompaniment mode')
        self._player.connect_to_proc(self._proc)
        self._msg_receiver = udp_pipe.UDPReceiver()
        self._follow_tempo = 0
        self._follow_tempo_ratio = 0
        self._live_display = None

        self._dump = self._config['dump']
        if self._dump:
            self._dump_dir = self._config['name']
            self._dmp_real_time = []
            self._dmp_play_time = []
            self._dmp_computed_tempo = []
            self._dmp_exec_time = []

    def loop(self):
        # wait for starting signal
        console.print('[green bold]Accompaniment module ready. Waiting for score following module...')
        while True:
            msg = self._msg_receiver()
            if msg is not None and msg['type'] == 'start':
                self._peer_start_time = msg['time']
                self._peer_score_tempo = msg['tempo'] / 60
                self._perf_tempo = self._peer_score_tempo
                self._follow_tempo = self._peer_score_tempo
                self._follow_tempo_ratio = 1
                break
            time.sleep(0.02)
        with Live(generate_run_info_table(), refresh_per_second=4, transient=True) as live:
            # share `live` within the class instance
            self._live_display = live
            self._player.loop()  # blocking
        # some cleaning work
        self._msg_receiver.close()
        if self._dump:
            shared_utils.check_dir('output', self._dump_dir)
            # write accompaniment result to file
            # calculate MIDI from recorded data points
            midi_ref = pretty_midi.PrettyMIDI(self._midi_path)
            midi_new = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))
            ax_time = [0] + [x - self._peer_start_time for x in self._dmp_real_time]
            ax_pos = [0] + [x for x in self._dmp_play_time]
            on_going_note = None
            for note in midi_ref.instruments[0].notes:
                for i in range(1, len(ax_pos)):
                    if ax_pos[i - 1] <= note.start <= ax_pos[i]:
                        on_going_note = (ax_time[i], note.pitch, note.velocity)
                    if on_going_note is not None and ax_pos[i - 1] <= note.end <= ax_pos[i]:
                        piano.notes.append(pretty_midi.Note(start=on_going_note[0],
                                                            end=ax_time[i],
                                                            pitch=on_going_note[1],
                                                            velocity=on_going_note[2]))
            midi_new.instruments.append(piano)
            midi_new.write(f"output/{self._dump_dir}/ac_output.mid")
            # copy original accompaniment MIDI file
            shutil.copyfile(self._config['acco_midi'], f"output/{self._dump_dir}/ac_origin.mid")
            # dump data points
            np.save(f"output/{self._dump_dir}/ac_real_time.npy", self._dmp_real_time)
            np.save(f"output/{self._dump_dir}/ac_play_time.npy", self._dmp_play_time)
            np.save(f"output/{self._dump_dir}/ac_computed_tempo.npy", self._dmp_computed_tempo)

    def _proc(self, a_time, a_output):
        # `a_` means `Audio`
        has_update = False
        while True:
            msg = self._msg_receiver()
            if msg is not None and msg['type'] == 'stop':
                a_output.kill()
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
            # estimated performance tempo in BPS, use weighted linear regression
            wls_model = sm.WLS(self._fax_pos, sm.add_constant(self._fax_time), weights=self._fax_conf)
            fit_params = wls_model.fit().params
            new_perf_tempo = fit_params[1]
            # if new_perf_tempo > 1.1 * self._perf_tempo:
            #     new_perf_tempo = 1.1 * self._perf_tempo
            # elif new_perf_tempo < 0.9 * self._perf_tempo:
            #     new_perf_tempo = 0.9 * self._perf_tempo
            self._perf_tempo = new_perf_tempo
            if self._perf_tempo > 0:
                current_pos = a_output.current_time * self._peer_score_tempo
                # the reason to use max() here is that UDP does not guarantee the order of arrival of packets
                self._follow_tempo = (fit_params[0] + (a_time - self._peer_start_time) * fit_params[1] + 4 - current_pos
                                      ) / (4 / self._perf_tempo - self._config['audio_input_latency'])
                self._follow_tempo = max(0, self._follow_tempo)  # in BPS
                # ratio compared to original performance tempo
                self._follow_tempo_ratio = self._follow_tempo / self._peer_score_tempo
            else:
                self._follow_tempo = 0
                self._follow_tempo_ratio = 0
            a_output.change_tempo_ratio(self._follow_tempo_ratio)
        # update live display
        self._live_display.update(generate_run_info_table((a_time - self._peer_start_time,
                                                           a_output.current_time,
                                                           a_output.total_time,
                                                           self._follow_tempo,
                                                           self._follow_tempo_ratio)))
        # dump data points
        if self._dump:
            self._dmp_real_time.append(a_time)
            self._dmp_play_time.append(a_output.current_time)
            if has_update:
                self._dmp_computed_tempo.append(self._follow_tempo)
            else:
                # use `-1` to mark no incoming message
                self._dmp_computed_tempo.append(-1)


if __name__ == '__main__':
    app = AutoAccompaniment(global_config.config)
    app.loop()
