import json
import math
import os
import sys
import traceback
import wave

import matplotlib
import numpy as np
import pretty_midi
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMainWindow, QApplication, QLabel, QPushButton, QLineEdit, \
    QHBoxLayout, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from utils import shared_utils, signal_processing

matplotlib.use('Qt5Agg')


def get_midi_tempo(project_name):
    midi_file = pretty_midi.PrettyMIDI(f"output/{project_name}/sf_origin.mid")
    return shared_utils.average(midi_file.get_tempo_changes()[1])


def get_config(project_name):
    with open(f"output/{project_name}/global_config.json", 'r') as fs:
        return json.loads(fs.read())


def parse_audio_file(config, project_name):
    with wave.open(f"output/{project_name}/audio_input.wav", 'rb') as wf:
        if config['perf_mode'] == 1:
            sample_rate = config['perf_sr']
        elif config['perf_mode'] == 0:
            # for WAV file input mode, the sample rate is determined by file
            if wf.getnchannels() != 1:
                raise Exception('unsupported channel number of WAV file')
            sample_rate = wf.getframerate()
        chunk_dur = config['perf_chunk'] / sample_rate
        sample_width = wf.getsampwidth()
        try:
            d_type = {
                1: np.int8,
                2: np.int16,
                4: np.int32,
                8: np.int64
            }[sample_width]
        except KeyError:
            raise Exception('unsupported sample width')
        bytes_read = wf.readframes(-1)
        array_amplitude = np.frombuffer(bytes_read, d_type)
        array_time = np.array([(i + 1) / sample_rate for i in range(len(array_amplitude))])
        return chunk_dur, array_time, array_amplitude


def update_vlines(h, x, ymin=None, ymax=None):
    seg_old = h.get_segments()
    if ymin is None:
        ymin = seg_old[0][0, 1]
    if ymax is None:
        ymax = seg_old[0][1, 1]

    seg_new = [np.array([[xx, ymin],
                         [xx, ymax]]) for xx in x]

    h.set_segments(seg_new)


class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=6, dpi=100, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class QPlotView(QWidget):
    def __init__(self, width=8, height=6, dpi=100, show_toolbar=False, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.canvas = MplCanvas(width=width, height=height, dpi=dpi)
        self.toolbar = NavigationToolbar(self.canvas, self)
        if show_toolbar:
            self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        w_width = 1900
        w_height = 950
        self.setWindowTitle('Dump Data Viewer')
        self.setFixedSize(QSize(w_width, w_height))

        # project selector
        self.layoutProjectSelector = QHBoxLayout()
        self.buttonListProjects = QPushButton()
        self.buttonListProjects.setText('List...')
        self.buttonListProjects.clicked.connect(self.on_button_list_projects_clicked)
        self.labelProjectName = QLabel()
        self.labelProjectName.setText('Project Name:')
        self.inputProjectName = QLineEdit()
        self.buttonOpenProject = QPushButton()
        self.buttonOpenProject.setText('Open')
        self.buttonOpenProject.clicked.connect(self.on_button_open_project_clicked)
        self.buttonShowDumpFileShapes = QPushButton()
        self.buttonShowDumpFileShapes.setText('Shapes...')
        self.buttonShowDumpFileShapes.clicked.connect(self.on_button_show_dump_file_shapes_clicked)
        self.buttonShowConfig = QPushButton()
        self.buttonShowConfig.setText('Config...')
        self.buttonShowConfig.clicked.connect(self.on_button_show_config_clicked)
        self.layoutProjectSelector.addWidget(self.buttonListProjects)
        self.layoutProjectSelector.addSpacing(20)
        self.layoutProjectSelector.addWidget(self.labelProjectName)
        self.layoutProjectSelector.addWidget(self.inputProjectName)
        self.layoutProjectSelector.addWidget(self.buttonOpenProject)
        self.layoutProjectSelector.addWidget(self.buttonShowDumpFileShapes)
        self.layoutProjectSelector.addWidget(self.buttonShowConfig)
        self.projectSelector = QWidget(self)
        self.projectSelector.setGeometry(0, 0, w_width // 2, 50)
        self.projectSelector.setLayout(self.layoutProjectSelector)

        # primary plot
        self.plotPrimary = QPlotView(show_toolbar=True, parent=self)
        self.plotPrimary.setGeometry(0, 50, w_width // 2, w_height - 50 * 2)
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        self.axProgress = self.plotPrimary.canvas.fig.add_subplot(gs[0])
        self.axAudioWaveform = self.plotPrimary.canvas.fig.add_subplot(gs[1], sharex=self.axProgress)
        self.axAudioFeatures = self.axAudioWaveform.twinx()

        # time selector
        self.layoutTimeSelector = QHBoxLayout()
        self.buttonBackward = QPushButton()
        self.buttonBackward.setText('<<')
        self.buttonBackward.setDisabled(True)
        self.buttonBackward.clicked.connect(self.on_button_backward_clicked)
        self.buttonPrevious = QPushButton()
        self.buttonPrevious.setText('<')
        self.buttonPrevious.setDisabled(True)
        self.buttonPrevious.clicked.connect(self.on_button_previous_clicked)
        self.buttonNext = QPushButton()
        self.buttonNext.setText('>')
        self.buttonNext.setDisabled(True)
        self.buttonNext.clicked.connect(self.on_button_next_clicked)
        self.buttonForward = QPushButton()
        self.buttonForward.setText('>>')
        self.buttonForward.setDisabled(True)
        self.buttonForward.clicked.connect(self.on_button_forward_clicked)
        self.labelCurIndex = QLabel()
        self.labelCurIndex.setText('Frame: ')
        self.labelCurIndexValue = QLabel()
        self.labelCurIndexValue.setText('-')
        self.labelSeparator = QLabel()
        self.labelSeparator.setText('/')
        self.labelTotalIndex = QLabel()
        self.labelTotalIndex.setText('-')
        self.labelSeparator2 = QLabel()
        self.labelSeparator2.setText('/')
        self.labelCurTime = QLabel()
        self.labelCurTime.setText('-')
        self.labelJumpTo = QLabel()
        self.labelJumpTo.setText('Jump to: ')
        self.inputJumpTo = QLineEdit()
        self.inputJumpTo.setDisabled(True)
        self.buttonJumpTo = QPushButton()
        self.buttonJumpTo.setText('Go')
        self.buttonJumpTo.setDisabled(True)
        self.buttonJumpTo.clicked.connect(self.on_button_go_clicked)
        self.layoutTimeSelector.addWidget(self.buttonBackward)
        self.layoutTimeSelector.addWidget(self.buttonPrevious)
        self.layoutTimeSelector.addWidget(self.buttonNext)
        self.layoutTimeSelector.addWidget(self.buttonForward)
        self.layoutTimeSelector.addSpacing(20)
        self.layoutTimeSelector.addWidget(self.labelCurIndex)
        self.layoutTimeSelector.addWidget(self.labelCurIndexValue)
        self.layoutTimeSelector.addWidget(self.labelSeparator)
        self.layoutTimeSelector.addWidget(self.labelTotalIndex)
        self.layoutTimeSelector.addWidget(self.labelSeparator2)
        self.layoutTimeSelector.addWidget(self.labelCurTime)
        self.layoutTimeSelector.addSpacing(20)
        self.layoutTimeSelector.addWidget(self.labelJumpTo)
        self.layoutTimeSelector.addWidget(self.inputJumpTo)
        self.layoutTimeSelector.addWidget(self.buttonJumpTo)
        self.layoutTimeSelector.setAlignment(Qt.AlignRight)
        self.timeSelector = QWidget(self)
        self.timeSelector.setGeometry(0, w_height - 50, w_width // 2, 50)
        self.timeSelector.setLayout(self.layoutTimeSelector)

        # secondary plot
        self.plotSecondary = QPlotView(show_toolbar=True, parent=self)
        self.plotSecondary.setGeometry(w_width // 2, 0, w_width // 2, w_height - 50)
        gs = GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 2])
        self.axIJ = self.plotSecondary.canvas.fig.add_subplot(gs[0])
        self.axI = self.plotSecondary.canvas.fig.add_subplot(gs[1])
        self.axPost = self.plotSecondary.canvas.fig.add_subplot(gs[2], sharex=self.axI)
        self.axV = self.plotSecondary.canvas.fig.add_subplot(gs[3], sharex=self.axI)
        self.axScore = self.plotSecondary.canvas.fig.add_subplot(gs[4], sharex=self.axI)

        # secondary zoom control
        self.layoutZoomControl = QHBoxLayout()
        self.labelCurZoom = QLabel()
        self.labelCurZoom.setText('Cur zoom: ')
        self.labelCurZoomValue = QLabel()
        self.labelCurZoomValue.setText('-')
        self.labelSetZoom = QLabel()
        self.labelSetZoom.setText('Set zoom: ')
        self.inputSetZoom = QLineEdit()
        self.inputSetZoom.setDisabled(True)
        self.buttonSetZoom = QPushButton()
        self.buttonSetZoom.setText('Set')
        self.buttonSetZoom.setDisabled(True)
        self.buttonSetZoom.clicked.connect(self.on_button_set_clicked)
        self.buttonClearZoom = QPushButton()
        self.buttonClearZoom.setText('Clear')
        self.buttonClearZoom.setDisabled(True)
        self.buttonClearZoom.clicked.connect(self.on_button_clear_clicked)
        self.layoutZoomControl.addWidget(self.labelCurZoom)
        self.layoutZoomControl.addWidget(self.labelCurZoomValue)
        self.layoutZoomControl.addSpacing(300)
        self.layoutZoomControl.addWidget(self.labelSetZoom)
        self.layoutZoomControl.addWidget(self.inputSetZoom)
        self.layoutZoomControl.addWidget(self.buttonSetZoom)
        self.layoutZoomControl.addWidget(self.buttonClearZoom)
        self.zoomControl = QWidget(self)
        self.zoomControl.setGeometry(w_width // 2, w_height - 50, w_width // 2, 50)
        self.zoomControl.setLayout(self.layoutZoomControl)

        # placeholders
        self.cfg = None
        self.cfg_chunk_dur = None
        self.audio_time = None
        self.audio_amplitude = None
        self.sf_score_tempo = None
        self.sf_real_time = None
        self.sf_cur_pos = None
        self.sf_sax_time = None
        self.sf_estimate_tempo = None
        self.sf_report_pos = None
        self.ac_real_time = None
        self.ac_play_time = None
        self.ac_computed_tempo = None
        self.sf_audio_pitch = None
        self.sf_audio_onset = None
        self.start_time = None
        self.sf_relative_time = None
        self.sf_pdf_ij = None
        self.sf_pdf_i = None
        self.sf_pdf_v = None
        self.sf_pdf_post = None
        self.resolution = None
        self.points_per_second = None
        self.window_size_ij_i_post = None
        self.window_size_v = None
        self.time_ij = None
        self.time_mask_i_post = None
        self.time_mask_v = None
        self.sf_sax_pitch = None
        self.sf_sax_onset = None
        self.cur_index = None
        self.secondary_zoom = None
        self.axProgressCurPosIndicator = None
        self.axAudioCurPosIndicator = None

    def on_button_list_projects_clicked(self):
        # noinspection PyBroadException
        try:
            info = os.listdir('output')
            QMessageBox.information(self, 'Project list', '\n'.join(info))
        except:
            QMessageBox.critical(self, 'Error while opening project...', traceback.format_exc())

    def on_button_open_project_clicked(self):
        # noinspection PyBroadException
        try:
            project_name = self.inputProjectName.text()
            self.load_project(project_name)
            # reset
            self.set_index(0)
            self.set_secondary_zoom(None)
            # plot
            self.plot_primary()
            self.plot_secondary()
            # enable buttons
            self.buttonBackward.setDisabled(False)
            self.buttonPrevious.setDisabled(False)
            self.buttonNext.setDisabled(False)
            self.buttonForward.setDisabled(False)
            self.inputJumpTo.setDisabled(False)
            self.buttonJumpTo.setDisabled(False)
            self.inputSetZoom.setDisabled(False)
            self.buttonSetZoom.setDisabled(False)
            self.buttonClearZoom.setDisabled(False)
        except FileNotFoundError:
            QMessageBox.critical(self, 'Error', 'No such project')
        except:
            QMessageBox.critical(self, 'Error while opening project...', traceback.format_exc())

    def on_button_show_dump_file_shapes_clicked(self):
        # noinspection PyBroadException
        try:
            project_name = self.inputProjectName.text()
            self.show_dump_file_shapes(project_name)
        except FileNotFoundError:
            QMessageBox.critical(self, 'Error', 'No such project')
        except:
            QMessageBox.critical(self, 'Error while showing dump file shapes...', traceback.format_exc())

    def on_button_show_config_clicked(self):
        # noinspection PyBroadException
        try:
            project_name = self.inputProjectName.text()
            self.show_config(project_name)
        except FileNotFoundError:
            QMessageBox.critical(self, 'Error', 'No such project')
        except:
            QMessageBox.critical(self, 'Error while showing config...', traceback.format_exc())

    def on_button_backward_clicked(self):
        self.move_index(-20)
        self.update_index_indicators()
        self.plot_secondary()

    def on_button_previous_clicked(self):
        self.move_index(-1)
        self.update_index_indicators()
        self.plot_secondary()

    def on_button_next_clicked(self):
        self.move_index(1)
        self.update_index_indicators()
        self.plot_secondary()

    def on_button_forward_clicked(self):
        self.move_index(20)
        self.update_index_indicators()
        self.plot_secondary()

    def on_button_go_clicked(self):
        try:
            self.set_index(int(self.inputJumpTo.text()) - 1)
            self.update_index_indicators()
            self.plot_secondary()
        except ValueError:
            QMessageBox.critical(self, 'Error', 'Invalid input')
        finally:
            self.inputJumpTo.setText('')

    def on_button_set_clicked(self):
        try:
            self.set_secondary_zoom(float(self.inputSetZoom.text()))
            self.plot_secondary()
        except ValueError:
            QMessageBox.critical(self, 'Error', 'Invalid input')
        finally:
            self.inputSetZoom.setText('')

    def on_button_clear_clicked(self):
        self.set_secondary_zoom(None)
        self.plot_secondary()

    def show_config(self, project_name):
        config_lines = [f"{k}: {v}" for k, v in get_config(project_name).items()]
        QMessageBox.information(self, 'Config', '\n'.join(config_lines))

    def show_dump_file_shapes(self, project_name):
        info = []
        for filename in os.listdir(f"output/{project_name}"):
            if filename.endswith('.npy'):
                file_path = f"output/{project_name}/{filename}"
                info.append(f"{filename}\t{str(np.load(file_path).shape)}")
        QMessageBox.information(self, 'Dump file shapes', '\n'.join(info))

    def load_project(self, project_name):
        self.cfg = get_config(project_name)
        self.cfg_chunk_dur, self.audio_time, self.audio_amplitude = parse_audio_file(self.cfg, project_name)
        self.sf_score_tempo = get_midi_tempo(project_name)
        self.sf_real_time = np.load(f"output/{project_name}/sf_real_time.npy")
        self.sf_cur_pos = np.load(f"output/{project_name}/sf_cur_pos.npy")
        self.sf_sax_time = np.load(f"output/{project_name}/sf_sax_time.npy")
        self.sf_estimate_tempo = np.load(f"output/{project_name}/sf_estimate_tempo.npy")
        self.sf_report_pos = np.load(f"output/{project_name}/sf_report_pos.npy")
        self.ac_real_time = np.load(f"output/{project_name}/ac_real_time.npy")
        self.ac_play_time = np.load(f"output/{project_name}/ac_play_time.npy")
        self.ac_computed_tempo = np.load(f"output/{project_name}/ac_computed_tempo.npy")
        self.sf_audio_pitch = np.load(f"output/{project_name}/sf_audio_pitch.npy")
        self.sf_audio_onset = np.load(f"output/{project_name}/sf_audio_onset.npy")
        self.start_time = min(self.sf_real_time[0], self.ac_real_time[0]) - self.cfg_chunk_dur
        self.sf_relative_time = np.array([t - self.start_time for t in self.sf_real_time])
        self.sf_pdf_ij = np.load(f"output/{project_name}/sf_pdf_ij.npy")
        self.sf_pdf_i = np.load(f"output/{project_name}/sf_pdf_i.npy")
        self.sf_pdf_v = np.load(f"output/{project_name}/sf_pdf_v.npy")
        self.sf_pdf_post = np.load(f"output/{project_name}/sf_pdf_post.npy")
        self.resolution = self.cfg_chunk_dur / self.cfg['resolution_multiple']
        self.points_per_second = math.ceil(1 / self.resolution)
        self.window_size_ij_i_post = math.ceil(self.cfg['window_ij'] / 2 * self.points_per_second) * 2
        self.window_size_v = math.ceil(self.cfg['window_v'] / 2 * self.points_per_second) * 2
        self.time_ij = np.array([self.resolution * i for i in range(self.window_size_ij_i_post)])
        self.time_mask_i_post = np.array([self.resolution * (- self.window_size_ij_i_post / 2 + i) for i in
                                          range(self.window_size_ij_i_post)])
        self.time_mask_v = np.array(
            [self.resolution * (- self.window_size_v / 2 + i) for i in range(self.window_size_v)])
        self.sf_sax_pitch = np.load(f"output/{project_name}/sf_sax_pitch.npy")
        self.sf_sax_onset = np.load(f"output/{project_name}/sf_sax_onset.npy")

    def set_index(self, value):
        self.cur_index = value
        self.cur_index = max(0, self.cur_index)
        self.cur_index = min(self.cur_index, len(self.sf_cur_pos) - 1)
        self.show_index()

    def move_index(self, shift):
        self.cur_index += shift
        self.cur_index = max(0, self.cur_index)
        self.cur_index = min(self.cur_index, len(self.sf_cur_pos) - 1)
        self.show_index()

    def show_index(self):
        self.labelCurIndexValue.setText(str(self.cur_index + 1))
        self.labelTotalIndex.setText(str(len(self.sf_cur_pos)))  # indeed, all arrays have same length
        self.labelCurTime.setText(f"{self.sf_relative_time[self.cur_index]:.3f}s")

    def set_secondary_zoom(self, value):
        self.secondary_zoom = value
        self.show_secondary_zoom()

    def show_secondary_zoom(self):
        if self.secondary_zoom is None:
            txt = '-'
        else:
            txt = f"{self.secondary_zoom:.4f}s"
        self.labelCurZoomValue.setText(txt)

    def update_index_indicators(self):
        update_vlines(self.axProgressCurPosIndicator, [self.sf_relative_time[self.cur_index]])
        update_vlines(self.axAudioCurPosIndicator, [self.sf_relative_time[self.cur_index]])
        self.plotPrimary.canvas.draw()

    def plot_primary(self, dot_size=20, edge_size=0.15, line_width=2):
        sf_score_time = np.array([self.sf_sax_time[pos] for pos in self.sf_cur_pos])
        sf_relative_time_for_estimate_tempo = self.sf_relative_time[self.sf_estimate_tempo != -1]
        sf_score_time_for_estimate_tempo = sf_score_time[self.sf_estimate_tempo != -1]
        sf_estimate_tempo_for_estimate_tempo = self.sf_estimate_tempo[self.sf_estimate_tempo != -1]
        sf_relative_time_for_report_pos = self.sf_relative_time[self.sf_report_pos != 0]
        sf_score_time_for_report_pos = sf_score_time[self.sf_report_pos != 0]
        ac_relative_time = np.array([t - self.start_time for t in self.ac_real_time])
        ac_relative_time_for_computed_tempo = ac_relative_time[self.ac_computed_tempo != -1]
        ac_play_time_for_computed_tempo = self.ac_play_time[self.ac_computed_tempo != -1]
        ac_computed_tempo_for_computed_tempo = self.ac_computed_tempo[self.ac_computed_tempo != -1]
        y_max = max(*sf_score_time, *self.ac_play_time)
        y_min = min(*sf_score_time, *self.ac_play_time)

        # draw progress axes
        # clear
        self.axProgress.clear()
        # draw sf progress line
        self.axProgress.plot(self.sf_relative_time, sf_score_time, color='tab:orange', label='sf', zorder=1)
        # draw sf tempo estimation
        self.axProgress.scatter(sf_relative_time_for_estimate_tempo, sf_score_time_for_estimate_tempo, s=dot_size,
                                color='tab:purple', label='sf estimate tempo', zorder=2)
        for i in range(len(sf_relative_time_for_estimate_tempo)):
            left_x = sf_relative_time_for_estimate_tempo[i] - edge_size
            left_y = sf_score_time_for_estimate_tempo[i] - edge_size * sf_estimate_tempo_for_estimate_tempo[
                i] / self.sf_score_tempo
            right_x = sf_relative_time_for_estimate_tempo[i] + edge_size
            right_y = sf_score_time_for_estimate_tempo[i] + edge_size * sf_estimate_tempo_for_estimate_tempo[
                i] / self.sf_score_tempo
            self.axProgress.plot([left_x, right_x], [left_y, right_y], color='tab:purple', linewidth=line_width,
                                 zorder=2)
        # draw sf position reporting
        self.axProgress.scatter(sf_relative_time_for_report_pos, sf_score_time_for_report_pos, s=dot_size,
                                color='tab:green', label='sf report pos', zorder=2)
        # draw ac progress line, simulate the audio output latency
        self.axProgress.plot(ac_relative_time + self.cfg['audio_input_latency'], self.ac_play_time, color='tab:blue',
                             label='ac (lat)', zorder=1)
        # draw ac computed tempo
        self.axProgress.scatter(ac_relative_time_for_computed_tempo, ac_play_time_for_computed_tempo, s=dot_size,
                                color='tab:red', label='ac computed tempo', zorder=2)
        for i in range(len(ac_relative_time_for_computed_tempo)):
            left_x = ac_relative_time_for_computed_tempo[i] - edge_size
            left_y = ac_play_time_for_computed_tempo[i] - edge_size * ac_computed_tempo_for_computed_tempo[i] / (
                    self.sf_score_tempo / 60)
            right_x = ac_relative_time_for_computed_tempo[i] + edge_size
            right_y = ac_play_time_for_computed_tempo[i] + edge_size * ac_computed_tempo_for_computed_tempo[i] / (
                    self.sf_score_tempo / 60)
            self.axProgress.plot([left_x, right_x], [left_y, right_y], color='tab:red', linewidth=line_width, zorder=2)
        # draw current position
        self.axProgressCurPosIndicator = self.axProgress.vlines([self.sf_relative_time[self.cur_index]], y_min, y_max,
                                                                color='tab:pink', zorder=3)
        # misc and config
        self.axProgress.legend(loc='lower right')
        self.axProgress.grid()
        self.axProgress.set_title('Progress')
        self.axProgress.set_xlabel('Real time /sec')
        self.axProgress.set_ylabel('Score time /sec')

        sf_audio_time = np.array([(i + 1) * self.cfg_chunk_dur for i in range(len(self.sf_audio_pitch))])
        sf_audio_pitch_filtered = np.array(
            [i if i != signal_processing.PitchProcessorCore.NO_PITCH else None for i in self.sf_audio_pitch])
        sf_audio_onset_list = sf_audio_time[self.sf_audio_onset == 1]

        # draw audio waveform axes
        # clear
        self.axAudioWaveform.clear()
        # draw audio amplitude
        self.axAudioWaveform.plot(self.audio_time, self.audio_amplitude, color='tab:olive')
        # misc and config
        self.axAudioWaveform.set_yticks([])  # hide y tick for waveform plot

        # draw audio feature axes
        # clear
        self.axAudioFeatures.clear()
        # draw audio pitch and onset
        self.axAudioFeatures.plot(sf_audio_time, sf_audio_pitch_filtered, color='tab:blue', label='pitch', zorder=2)
        self.axAudioFeatures.vlines(sf_audio_onset_list, 0, 11, color='tab:red', label='onset', zorder=1)
        # draw current position
        self.axAudioCurPosIndicator = self.axAudioFeatures.vlines([self.sf_relative_time[self.cur_index]], 0, 11,
                                                                  color='tab:pink', zorder=3)
        # misc and config
        self.axAudioFeatures.legend(loc='upper right')
        self.axAudioFeatures.grid()
        # set key notes as ticks
        self.axAudioFeatures.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.axAudioFeatures.set_yticklabels(['C', None, 'D', None, 'E', 'F', None, 'G', None, 'A', None, 'B'],
                                             rotation=45)
        # by default y-ticks are on right side, move them to left side
        self.axAudioFeatures.yaxis.set_label_position('left')
        self.axAudioFeatures.yaxis.tick_left()
        self.axAudioFeatures.set_title('Audio')
        self.axAudioFeatures.set_xlabel('Real time /sec')
        self.axAudioFeatures.set_ylabel('Pitch')

        # render
        self.plotPrimary.canvas.fig.tight_layout()
        self.plotPrimary.canvas.draw()

    def plot_secondary(self):
        # IJ
        self.axIJ.clear()
        self.axIJ.plot(self.time_ij, self.sf_pdf_ij[self.cur_index])
        self.axIJ.set_title('$f(I-J|D)$')

        prev_time = self.sf_sax_time[self.sf_cur_pos[self.cur_index - 1]] if self.cur_index > 0 else 0
        cur_pos = self.sf_cur_pos[self.cur_index]
        cur_time = self.sf_sax_time[cur_pos]

        # I
        self.axI.clear()
        y = self.sf_pdf_i[self.cur_index]
        self.axI.plot(self.time_mask_i_post + cur_time, y, color='tab:blue')
        self.axI.vlines([cur_time], 0, max(y), color='tab:pink')
        self.axI.vlines([prev_time], 0, max(y), color='tab:gray')
        self.axI.set_title('$f(I|D)$')

        # posterior
        self.axPost.clear()
        y = self.sf_pdf_post[self.cur_index]
        self.axPost.plot(self.time_mask_i_post + cur_time, y, color='tab:blue')
        self.axPost.vlines([cur_time], 0, max(y), color='tab:pink')
        self.axPost.vlines([prev_time], 0, max(y), color='tab:gray')
        self.axPost.set_title('Posterior')

        # V
        self.axV.clear()
        y = self.sf_pdf_v[self.cur_index]
        self.axV.plot(self.time_mask_v + cur_time, y, color='tab:blue')
        self.axV.vlines([cur_time], 0, max(y), color='tab:pink')
        self.axV.vlines([prev_time], 0, max(y), color='tab:gray')
        self.axV.set_title('$f(V|I)$')

        # score
        window_size = max(self.window_size_ij_i_post, self.window_size_v)
        left = max(0, cur_pos - window_size // 2)
        right = min(cur_pos + window_size // 2, len(self.sf_sax_time))
        sf_sax_time_slice = np.array(self.sf_sax_time[left: right])
        sf_sax_pitch_slice = self.sf_sax_pitch[left:right]
        sf_sax_pitch_slice_filtered = [i if i != signal_processing.PitchProcessorCore.NO_PITCH else None for i in
                                       sf_sax_pitch_slice]
        sf_sax_onset_slice = np.array(self.sf_sax_onset[left:right])
        sf_sax_onset_slice_list = sf_sax_time_slice[sf_sax_onset_slice == 1]
        self.axScore.clear()
        self.axScore.plot(sf_sax_time_slice, sf_sax_pitch_slice_filtered, color='tab:blue', label='pitch')
        self.axScore.vlines(sf_sax_onset_slice_list, 0, 11, color='tab:red', label='onset')
        self.axScore.vlines([cur_time], 0, 11, color='tab:pink')
        self.axScore.vlines([prev_time], 0, 11, color='tab:gray')
        self.axScore.set_title('Score')
        self.axScore.legend(loc='upper right')
        self.axScore.grid()
        # set key notes as ticks
        self.axScore.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.axScore.set_yticklabels(['C', None, 'D', None, 'E', 'F', None, 'G', None, 'A', None, 'B'],
                                     rotation=45)

        # set x-limit if zoom is set
        if self.secondary_zoom is not None:
            self.axIJ.set_xlim([-0.1, self.secondary_zoom + 0.1])
            self.axI.set_xlim([cur_time - self.secondary_zoom / 2, cur_time + self.secondary_zoom / 2])

        # render
        self.plotSecondary.canvas.fig.tight_layout()
        self.plotSecondary.canvas.draw()


app = QApplication(sys.argv)
win = MainWindow()
win.show()
app.exec_()
