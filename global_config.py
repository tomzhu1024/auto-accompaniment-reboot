config = {
    'name': 'debug',

    # Input mode of performance
    #   > `0` - wave file
    #   > `1` - microphone
    'perf_mode': 1,

    # File path of performance wave file
    #   * Takes effects only when `perf_mode` is set to `0`
    'perf_audio': '',

    # Sample rate of performance input
    #   * Takes effects only when `perf_mode` is set to `1`
    #   * Advanced options
    'perf_sr': 44100,

    # Number of samples processed in each iteration
    #   * Advanced options
    'perf_chunk': 2048,

    # Number of discrete points that the probability density function keeps within the duration of one audio chunk
    #   * Advanced options
    'resolution_scalar': 4,

    # The size of the window of the f-IJ-given-D density function
    #   * Advanced options
    'window_ij': 25,

    # The size of the window of the f-V-given-D density function
    #   * Advanced options
    'window_v': 1,

    # The size of the gate of the posterior density function
    #   * Advanced options
    'gate_post': 5,

    # File path of performance MIDI file
    'score_midi': 'resources/pop909_melody/063.mid',

    # File path of accompaniment MIDI file
    'acco_midi': 'resources/pop909_acco/063.mid',

    # Output mode of accompaniment
    #   > `0` - Virtual MIDI synthesizer
    #   > `1` - External MIDI synthesizer
    'acco_mode': 0,

    # Name of External MIDI Device
    #   * Takes effect only when `acco_mode` is set to `1`
    'acco_device': 'CASIO USB-MIDI',

    # Save verbose information to file for debugging
    'dump': True,

    # The time to stop before the score ends, set to `0` to disable it
    #   * Takes effect only when `dump` is set to `True`
    'trunc_time': 90
}
