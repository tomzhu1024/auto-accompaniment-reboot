import wave

import matplotlib.pyplot as plt
import numpy as np
import pyaudio


def plot_sf_perf(config):
    perf = np.load('output/%s/perf.npy' % config['name'])
    chunk_dur = config['perf_chunk'] / config['perf_sr']
    start = 1 * chunk_dur
    end = (len(perf) + 1) * chunk_dur
    ax_time = np.arange(start, end, chunk_dur)
    plt.plot(ax_time, perf * 1000, '.', color='deepskyblue')
    plt.title('Performance Statistic')
    plt.xlabel('Time - s')
    plt.ylabel('Analysis Interval - ms')
    plt.hlines(config['perf_chunk'] / config['perf_sr'] * 1000, start, end, color='darkred')
    plt.show()


def plot_sf_core(config):
    p_ij = np.load('output/%s/ij.npy' % config['name'])
    p_i = np.load('output/%s/i.npy' % config['name'])
    p_v = np.load('output/%s/v.npy' % config['name'])
    p_post = np.load('output/%s/post.npy' % config['name'])
    assert len(p_ij) == len(p_i) == len(p_v)
    wf = wave.open(config['perf_audio'], 'rb')
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
    i = 0
    wf.setpos(i * config['perf_chunk'])
    bytes_read = wf.readframes(config['perf_chunk'])
    while i < len(p_ij) and bytes_read:
        stream.write(bytes_read)
        plt.subplot(411)
        plt.plot(p_ij[i])
        plt.subplot(412)
        plt.plot(p_i[i])
        plt.subplot(413)
        plt.plot(p_v[i])
        plt.subplot(414)
        plt.plot(p_post[i])
        plt.vlines(50, 0, 1, color='red')
        plt.show()
        i += 1
        bytes_read = wf.readframes(config['perf_chunk'])


if __name__ == '__main__':
    test_config = {
        'name': 'debug_2',
        'perf_audio': 'audio/audio3.wav',
        'perf_chunk': 1024,
        'perf_sr': 44100,
        'score_midi': 'midi/midi3.mid',
        'score_resolution': 0.01
    }
    plot_sf_perf(test_config)
    plot_sf_core(test_config)
