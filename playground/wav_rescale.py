import math
import os

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from pydub import AudioSegment

if __name__ == '__main__':
    """Configuration"""
    filename = 'audio3'
    interval = 30
    scaling_factors = [
        [0.7, 1.3, 0.7, 1.3, 0.7, 1.3],
        [1.3, 0.7, 1.3, 0.7, 1.3, 0.7],
    ]
    """End of configuration"""

    assert len(scaling_factors) > 0, 'at least one scaling factors combination should be provided'
    assert len(set([len(x) for x in scaling_factors])) == 1, 'all scaling factors combinations should have a same size'

    audio = AudioSegment.from_file(f'{filename}.wav', format='wav')
    duration = audio.duration_seconds
    chunk_num = math.ceil(duration / interval)

    print('chunk duration:', interval)
    print('audio duration:', duration)
    print('# of chunks:', chunk_num)
    assert len(scaling_factors[0]) == chunk_num, \
        'the size of the scaling factors should be consistent with the audio duration'

    if not os.path.exists('temp'):
        os.mkdir('temp')
    else:
        assert os.path.isdir('temp'), './temp exists but is not a directory'

    for scaling_factor in scaling_factors:
        print('using scaling factors:', scaling_factor, end=' ')
        for i in range(chunk_num):
            tmp = audio[i * interval * 1000: (i + 1) * interval * 1000]
            tmp.export(f'temp/chunk_{str(i)}.wav', format='wav')
            with WavReader(f'temp/chunk_{str(i)}.wav') as reader:
                with WavWriter(f'temp/chunk_{str(i)}_rescaled.wav', reader.channels, reader.samplerate) as writer:
                    tsm = phasevocoder(reader.channels, speed=scaling_factor[i])
                    tsm.run(reader, writer)
            print('*', end=' ')
        print('')
        rescaled_audios = [AudioSegment.from_file(f'temp/chunk_{str(x)}_rescaled.wav', format='wav')
                           for x in range(chunk_num)]
        sum(rescaled_audios).export(f'{filename}_{"_".join([str(x) for x in scaling_factor])}.wav')
