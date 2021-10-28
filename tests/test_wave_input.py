from unittest import TestCase

import time

from utils.audio_io import WaveFileInput

STOP_TIME = 20

config = {
    'perf_audio': 'test.wav',
    'perf_chunk': 2048
}
wf_input = WaveFileInput(config)


def callback(read_time, prev_time, data, audio_input: WaveFileInput):
    # raise Exception('test exception')
    print(f"Read time: {read_time:.2f}\tPrev time: {prev_time:.2f}\t"
          f"Elapsed time: {read_time - audio_input.start_time:.2f}\tFrame length: {len(data)}")
    if read_time - audio_input.start_time > STOP_TIME:
        audio_input.kill()
    time.sleep(0.03)


wf_input.connect_to_proc(callback)
print(f"Start time: {time.time():.2f}")
wf_input.loop()


