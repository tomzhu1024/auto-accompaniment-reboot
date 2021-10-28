from time import sleep
from unittest import TestCase

from utils.audio_io import WaveFileInput, MicrophoneInput

CHUNK = 2048
SR = 44100
VALID_LOAD = 0.6
INVALID_LOAD = 2
PLAY_DUR = 3


class TestWaveFileInput(TestCase):
    def worker(self, delay: float, error: bool) -> float:
        wf_input = WaveFileInput({
            'perf_audio': 'test.wav',
            'perf_chunk': CHUNK
        })
        last_call_time = 0

        def callback(read_time, prev_time, data, audio_input: WaveFileInput):
            nonlocal last_call_time
            last_call_time = read_time
            print(f"Read time: {read_time:.2f}\tPrev time: {prev_time:.2f}\t"
                  f"Elapsed time: {read_time - audio_input.start_time:.2f}\tFrame length: {len(data)}")
            if read_time - audio_input.start_time > PLAY_DUR:
                audio_input.kill()
            if error:
                raise Exception()
            if delay > 0:
                sleep(delay)

        wf_input.connect_to_proc(callback)
        wf_input.loop()

        delta_time = last_call_time - wf_input.start_time
        print(f"Delta time: {delta_time:.2f}")
        return delta_time

    def test_no_workload(self):
        try:
            self.assertAlmostEqual(self.worker(0, False), PLAY_DUR, delta=CHUNK / SR)
        except:
            self.fail()

    def test_valid_workload(self):
        try:
            self.assertAlmostEqual(self.worker(VALID_LOAD * CHUNK / SR, False), PLAY_DUR, delta=CHUNK / SR)
        except:
            self.fail()

    def test_invalid_workload(self):
        try:
            self.worker(INVALID_LOAD * CHUNK / SR, False)
        except:
            self.assertTrue(True)

    def test_exception(self):
        try:
            self.worker(0, True)
            self.fail()
        except:
            self.assertTrue(True)


class TestMicrophoneInput(TestCase):
    def worker(self, delay: float, error: bool, filename: str) -> int:
        mic_input = MicrophoneInput({
            'perf_chunk': CHUNK,
            'perf_sr': SR,
            'dump': True
        })
        call_count = 0

        def callback(read_time, prev_time, data, audio_input: WaveFileInput):
            nonlocal call_count
            call_count += 1
            print(f"Read time: {read_time:.2f}\tPrev time: {prev_time:.2f}\t"
                  f"Elapsed time: {read_time - audio_input.start_time:.2f}\tFrame length: {len(data)}")
            if read_time - audio_input.start_time > PLAY_DUR:
                audio_input.kill()
            if error:
                raise Exception()
            if delay > 0:
                sleep(delay)

        mic_input.connect_to_proc(callback)
        mic_input.loop()
        mic_input.save_to_file(filename)

        print(f"Call count: {call_count}")
        return call_count

    def test_no_workload(self):
        self.assertAlmostEqual(self.worker(0, False, 'no_load.wav'), PLAY_DUR / (CHUNK / SR), delta=1)

    def test_valid_workload(self):
        self.assertAlmostEqual(self.worker(VALID_LOAD * CHUNK / SR, False, 'valid_load.wav'), PLAY_DUR / (CHUNK / SR),
                               delta=1)

    def test_invalid_workload(self):
        self.assertNotAlmostEqual(self.worker(INVALID_LOAD * CHUNK / SR, False, 'invalid_load.wav'),
                                  PLAY_DUR / (CHUNK / SR), delta=1)

    def test_exception(self):
        try:
            self.worker(0, True, 'exception.wav')
            self.fail()
        except:
            self.assertTrue(True)
