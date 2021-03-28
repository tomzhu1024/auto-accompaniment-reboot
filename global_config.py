config = {
    'name': 'expectationtest-006',

    # Input mode of performance
    #   0 - wave file
    #   1 - microphone
    'perf_mode': 0,

    # File path of performance wave file
    #   takes effects only when perf_mode is set to 0
    'perf_audio': 'output/expectationtest-005/audio_mic.wav',

    # Sample rate of performance input
    #   takes effects only when perf_mode is set to 1
    'perf_sr': 44100,

    # Number of samples processed in each iteration
    'perf_chunk': 2048,

    # File path of score MIDI file (score)
    'score_midi': 'resources/pop909_melody/019.mid',

    # Output mode of accompaniment
    #   0 - Virtual MIDI synthesizer
    #   1 - External MIDI synthesizer
    'acco_mode': 0,

    # Name of External MIDI Device
    #   takes effect only when acco_mode is set to 1
    'acco_device': 'CASIO USB-MIDI',

    # File path of accompaniment MIDI file
    'acco_midi': 'resources/pop909_acco/019.mid',

    # More output for debug purpose
    'dump': True,

    # Stop earlier than the score ends, set to non-positive to disable
    'trunc_time': 70
}
