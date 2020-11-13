config = {
    'name': '20201113-play',

    # Input mode of performance
    #   0 - wave file
    #   1 - microphone
    'perf_mode': 1,

    # File path of performance wave file
    #   takes effects only when perf_mode is set to 0
    'perf_audio': 'resources/audio3.wav',

    # Sample rate of performance input
    #   takes effects only when perf_mode is set to 1
    #   when perf_mode is set to 0, this value will be overwritten by wave file's sample rate
    'perf_sr': 44100,

    # Number of samples processed in each iteration
    'perf_chunk': 1024,

    # File path of score MIDI file (score)
    'score_midi': 'resources/midi4.mid',

    # Output mode of accompaniment
    #   0 - Virtual MIDI synthesizer
    #   1 - External MIDI synthesizer
    'acco_mode': 1,

    # File path of accompaniment MIDI file
    #   takes effect only when acco_mode is set to 0
    'acco_midi': 'resources/midi4.mid',

    # More output for debug purpose
    'dump': True,

    # Stop earlier than the score ends, set to non-positive to disable
    'trunc_time': 90
}
