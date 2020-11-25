import matplotlib.pyplot as plt
import numpy as np
import math


def normalize(array):
    return np.true_divide(array, np.sum(array))


def compute_f_i_j_given_d(time_axis, d, score_tempo, estimated_tempo):
    rate_ratio = estimated_tempo / score_tempo if estimated_tempo > 0 else score_tempo / 1e-5
    sigma_square = math.log(1 / (10 * d) + 1)
    sigma = math.sqrt(sigma_square)
    a = np.true_divide(1, np.multiply(time_axis, sigma * math.sqrt(2 * math.pi)), where=time_axis != 0)
    b = np.add(np.log(time_axis, where=time_axis != 0), 0.5 * sigma_square - math.log(rate_ratio * d))
    b = np.exp(np.true_divide(-np.square(b), 2 * sigma_square))
    f_i_j_given_d = np.multiply(a, b)
    # remove the possible np.nan element in the beginning, otherwise normalization will fail
    f_i_j_given_d[time_axis == 0] = 0
    f_i_j_given_d = normalize(f_i_j_given_d)
    return f_i_j_given_d


def compute_f_i_given_d(f_source, f_i_j_given_d, cur_pos, axis_length):
    # avoid overflow
    left = max(0, cur_pos - 1000)
    right = min(cur_pos + 1000, axis_length)
    f_i_given_d = np.zeros(axis_length)
    f_source_w = f_source[left:right]
    f_i_j_given_d_w = f_i_j_given_d[:right - left]
    f_i_given_d_w = np.convolve(f_source_w, f_i_j_given_d_w)
    f_i_given_d_w = f_i_given_d_w[:right - left]  # slice to window size
    f_i_given_d[left:right] = f_i_given_d_w
    f_i_given_d = normalize(f_i_given_d)
    return f_i_given_d


if __name__ == '__main__':
    ta = np.arange(0, 180, 0.01)
    ogn_tempo = 108
    est_tempo = 108
    delta_time = 1024/44100
    f_source = np.zeros(len(ta))
    cur_pos = 0
    f_source[cur_pos] = 1.0
    for i in range(1000):
        t = compute_f_i_j_given_d(ta, delta_time, ogn_tempo, est_tempo)
        f_source = compute_f_i_given_d(f_source, t, cur_pos, len(ta))
        # plt.plot(f_source[:100])
        # plt.show()
        cur_pos = np.argmax(f_source)
        # print(cur_pos)
    print(ta[cur_pos])
    # plt.legend()
    # plt.show()
