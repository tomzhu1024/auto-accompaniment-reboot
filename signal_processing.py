import aubio
import numpy as np


# TODO: rewrite this to better model
class PitchProcessorCore:
    NO_PITCH = -1.0
    DEPTH = 2
    WEIGHT = (0.25, 0.25)

    def __init__(self):
        self._edge = 0.5
        self._prev_pitches = None  # 上一个
        self._cur_pitch = PitchProcessorCore.NO_PITCH  # 当前
        self._cur_pitch_reverse = PitchProcessorCore.NO_PITCH  # 当前反转
        self._result = PitchProcessorCore.NO_PITCH  # 结果
        self._result_reverse = PitchProcessorCore.NO_PITCH  # 结果反转

    def __call__(self, value):
        if value != PitchProcessorCore.NO_PITCH:
            if self._prev_pitches is None:
                self._prev_pitches = np.full(PitchProcessorCore.DEPTH, value % 12)
            else:
                self._prev_pitches = np.roll(self._prev_pitches, 1)
                self._prev_pitches[0] = self._cur_pitch  # 保存上一个
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
        self._chunk_concat = 3  # 聚合的区块数量，聚合后数据更加稳定

        self._pitch_detector = aubio.pitch(method='yin',
                                           buf_size=self._chunk * self._chunk_concat,
                                           hop_size=self._chunk * self._chunk_concat,
                                           samplerate=self._samp_rate)
        self._pitch_detector.set_unit('midi')
        self._pitch_detector.set_tolerance(0.75)  # 音高识别阈值，越低越灵敏，但会产生大量低置信数据

        self._c_data = np.zeros(self._chunk * self._chunk_concat, dtype=np.float32)  # 多个区块聚合后的音频信号

    def __call__(self, data):
        # 新区块加入聚合
        self._c_data = np.roll(self._c_data, -self._chunk)  # 左移一个区块的距离
        self._c_data[-self._chunk:] = data  # 新区块覆盖右侧
        # 检测音高
        pitch = self._pitch_detector(self._c_data)[0]
        # 如果音高识别的可信度低于阈值，pitch的值为0.0
        # 但依然会有过低或过高的数据，需要再次过滤
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

        self._result = False  # 用布尔值表示onset

    def __call__(self, data):
        self._result = bool(self._onset_detector(data))
        return self._result

    @property
    def result(self):
        return self._result


if __name__ == '__main__':
    test_ppc = PitchProcessorCore()
    test_np = PitchProcessorCore.NO_PITCH
    test_seq = [test_np, test_np, test_np, test_np, 0.4, 0.4, 0.4, 0.4, test_np, test_np, 5, 5, 5]
    for i in test_seq:
        print('process', i, 'return', test_ppc(i), 'reversed', test_ppc.result(11.9))
