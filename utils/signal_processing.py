import aubio
import numpy as np


class PitchProcessorCore:
    NO_PITCH = -1.0
    DEPTH = 1
    WEIGHT = 0.5

    def __init__(self):
        self._edge = 0.5
        self._prev_pitches = None
        self._cur_pitch = PitchProcessorCore.NO_PITCH
        self._cur_pitch_reverse = PitchProcessorCore.NO_PITCH
        self._result = PitchProcessorCore.NO_PITCH
        self._result_reverse = PitchProcessorCore.NO_PITCH

    def __call__(self, value):
        if value != PitchProcessorCore.NO_PITCH:
            if self._prev_pitches is None:
                self._prev_pitches = np.full(PitchProcessorCore.DEPTH, value % 12)
            else:
                self._prev_pitches = np.roll(self._prev_pitches, 1)
                self._prev_pitches[0] = self._cur_pitch  # save previous one
            self._cur_pitch = value % 12
            self._result = self._cur_pitch - np.dot(self._prev_pitches, PitchProcessorCore.WEIGHT)

            if self._cur_pitch < self._edge:
                self._cur_pitch_reverse = self._cur_pitch + 12
                self._result_reverse = self._cur_pitch_reverse - np.dot(self._prev_pitches, PitchProcessorCore.WEIGHT)
            elif self._cur_pitch > 12 - self._edge:
                self._cur_pitch_reverse = self._cur_pitch - 12
                self._result_reverse = self._cur_pitch_reverse - np.dot(self._prev_pitches, PitchProcessorCore.WEIGHT)
            else:
                self._cur_pitch_reverse = PitchProcessorCore.NO_PITCH
                self._result_reverse = PitchProcessorCore.NO_PITCH
        else:
            self._cur_pitch = PitchProcessorCore.NO_PITCH
            self._result = PitchProcessorCore.NO_PITCH

            self._cur_pitch_reverse = PitchProcessorCore.NO_PITCH
            self._result_reverse = PitchProcessorCore.NO_PITCH
        return self._result

    def result(self, reference=None):
        if reference and abs(self._cur_pitch - reference) > abs(self._cur_pitch_reverse - reference):
            return self._result_reverse
        else:
            return self._result


class PitchProcessor(PitchProcessorCore):
    def __init__(self, config):
        super().__init__()

        self._chunk = config['perf_chunk']
        self._samp_rate = config['perf_sr']
        self._chunk_concat = 3  # aggregate several chunks to make output more stable

        self._pitch_detector = aubio.pitch(method='yin',
                                           buf_size=self._chunk * self._chunk_concat,
                                           hop_size=self._chunk * self._chunk_concat,
                                           samplerate=self._samp_rate)
        self._pitch_detector.set_unit('midi')
        self._pitch_detector.set_tolerance(0.75)

        self._c_data = np.zeros(self._chunk * self._chunk_concat, dtype=np.float32)  # aggregated chunks

    def __call__(self, data):
        # add new chunk to aggregation
        self._c_data = np.roll(self._c_data, -self._chunk)  # left shift one chunk
        self._c_data[-self._chunk:] = data  # overwritten the right side
        # detect pitch
        pitch = self._pitch_detector(self._c_data)[0]
        # if detected pitch confidence is below threshold, 0.0 will be returned
        # however, there are still some low-confidence data, another filtering is required
        if 40 < pitch < 84:
            super().__call__(pitch)
        else:
            super().__call__(PitchProcessorCore.NO_PITCH)
        return self._result


class OnsetProcessor:
    def __init__(self, config):
        self._chunk = config['perf_chunk']
        self._samp_rate = config['perf_sr']

        self._onset_detector = aubio.onset('default', self._chunk, self._chunk, self._samp_rate)

        self._result = False  # use boolean to represent onset

    def __call__(self, data):
        self._result = bool(self._onset_detector(data))
        return self._result

    @property
    def result(self):
        return self._result
