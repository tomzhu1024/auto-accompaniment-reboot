config = {
    # Name of the profile
    'name': 'test',

    # Input mode of performance
    #   > `0` - wave file
    #   > `1` - microphone
    'perf_mode': 1,

    # File path of performance wave file
    #   * Takes effects only when `perf_mode` is set to `0`
    'perf_audio': 'resources/star.wav',

    # Sample rate of performance input
    #   * Takes effects only when `perf_mode` is set to `1`
    #   * Advanced options
    'perf_sr': 44100,

    # Number of samples processed in each iteration
    #   * Advanced options
    'perf_chunk': 4096,

    # Number of discrete points that the probability density function keeps within the duration of one audio chunk
    #   * Advanced options
    'resolution_multiple': 4,

    # The size of the window of the f-IJ-given-D density function
    #   * Advanced options
    'window_ij': 25,

    # The size of the window of the f-V-given-D density function
    #   * Advanced options
    'window_v': 1.5,

    # The size of the gate of the posterior density function
    #   * Advanced options
    'gate_post': 25,

    # The time to wait between each tempo re-estimation
    #   * Advanced options
    'tempo_estimate_interval': 1,

    # The time to wait between each IPC message
    #   * Advanced options
    'pos_report_interval': 2,

    # The number of historical points the regression model takes consideration into
    #   * Advanced options
    'regression_depth': 4,

    # The input latency caused by the audio buffer
    #   * Advanced options
    'audio_input_latency': 0.2,

    # The interval in beat between beat-wise regularization
    #   # Advanced options
    'beat_reg_dist': 0.5,

    # File path of performance MIDI file
    'score_midi': 'resources/pop909/pop909_melody/710.mid',

    # File path of accompaniment MIDI file
    'acco_midi': 'resources/pop909/pop909_acco/710.mid',

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
    'trunc_time': 0
}
