import json

import shared_utils

config = {
    'name': 'debug',

    # Input mode of performance
    #   0 - wave file
    #   1 - microphone
    'perf_mode': 0,

    # File path of performance wave file
    #   takes effects only when perf_mode is set to 0
    'perf_audio': 'audio/audio3.wav',

    # Sample rate of performance input
    #   takes effects only when perf_mode is set to 1
    #   when perf_mode is set to 0, this value will be overwritten by wave file's sample rate
    'perf_sr': 44100,

    # Number of samples processed in each iteration
    'perf_chunk': 1024,

    # File path of score MIDI file (score)
    'score_midi': 'midi/midi3.mid',

    # Output mode of accompaniment
    #   0 - MIDI
    #   1 - wave file
    'acco_mode': 0,

    # File path of accompaniment MIDI file
    #   takes effect only when acco_mode is set to 0
    'acco_midi': 'midi/midi3.mid',

    # File path of accompaniment wave file
    #   takes effect only when acco_mode is set to 1
    'acco_audio': 'audio/audio3.wav',

    # More output for debug purpose
    'dump': True
}

if __name__ == '__main__':
    config_name = 'default'
    filename = 'config/%s.json' % config_name
    shared_utils.check_dir('config')
    with open(filename, 'w') as fs:
        fs.write(json.dumps(config))
    print('configuration saved to %s' % filename)
