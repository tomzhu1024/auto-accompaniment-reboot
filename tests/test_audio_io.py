from time import time, sleep
from unittest import TestCase

from utils.audio_io import WaveFileInput

CHUNK = 2048
SR = 44100
VALID_LOAD = 0.6
INVALID_LOAD = 2
PLAY_DUR = 3


class TestWaveFileInput(TestCase):
    def test_no_workload(self):
        try:
            wf_input = WaveFileInput({
                'perf_audio': 'test.wav',
                'perf_chunk': CHUNK
            })
            after_time = 0

            def callback(read_time, prev_time, data, audio_input: WaveFileInput):
                nonlocal after_time
                after_time = read_time
                print(f"Read time: {read_time:.2f}\tPrev time: {prev_time:.2f}\t"
                      f"Elapsed time: {read_time - audio_input.start_time:.2f}\tFrame length: {len(data)}")
                if read_time - audio_input.start_time > PLAY_DUR:
                    audio_input.kill()

            wf_input.connect_to_proc(callback)
            before_time = time()
            print(f"Before start time: {before_time:.2f}")
            wf_input.loop()
            self.assertTrue(abs(after_time - before_time - PLAY_DUR) < CHUNK / SR)
        except:
            self.fail()

    def test_valid_workload(self):
        try:
            wf_input = WaveFileInput({
                'perf_audio': 'test.wav',
                'perf_chunk': CHUNK
            })
            after_time = 0

            def callback(read_time, prev_time, data, audio_input: WaveFileInput):
                nonlocal after_time
                after_time = read_time
                print(f"Read time: {read_time:.2f}\tPrev time: {prev_time:.2f}\t"
                      f"Elapsed time: {read_time - audio_input.start_time:.2f}\tFrame length: {len(data)}")
                if read_time - audio_input.start_time > PLAY_DUR:
                    audio_input.kill()
                sleep(CHUNK / SR * VALID_LOAD)

            wf_input.connect_to_proc(callback)
            before_time = time()
            print(f"Before start time: {before_time:.2f}")
            wf_input.loop()
            self.assertTrue(abs(after_time - before_time - PLAY_DUR) < CHUNK / SR)
        except:
            self.fail()

    def test_invalid_workload(self):
        try:
            wf_input = WaveFileInput({
                'perf_audio': 'test.wav',
                'perf_chunk': CHUNK
            })

            def callback(read_time, prev_time, data, audio_input: WaveFileInput):
                print(f"Read time: {read_time:.2f}\tPrev time: {prev_time:.2f}\t"
                      f"Elapsed time: {read_time - audio_input.start_time:.2f}\tFrame length: {len(data)}")
                if read_time - audio_input.start_time > PLAY_DUR:
                    audio_input.kill()
                sleep(CHUNK / SR * INVALID_LOAD)

            wf_input.connect_to_proc(callback)
            before_time = time()
            print(f"Before start time: {before_time:.2f}")
            wf_input.loop()
            self.fail()
        except:
            self.assertTrue(True)

    def test_exception(self):
        try:
            wf_input = WaveFileInput({
                'perf_audio': 'test.wav',
                'perf_chunk': CHUNK
            })

            def callback(read_time, prev_time, data, audio_input: WaveFileInput):
                print(f"Read time: {read_time:.2f}\tPrev time: {prev_time:.2f}\t"
                      f"Elapsed time: {read_time - audio_input.start_time:.2f}\tFrame length: {len(data)}")
                if read_time - audio_input.start_time > PLAY_DUR:
                    audio_input.kill()
                raise Exception('test exception')

            wf_input.connect_to_proc(callback)
            before_time = time()
            print(f"Before start time: {before_time:.2f}")
            wf_input.loop()
            self.fail()
        except:
            self.assertTrue(True)
