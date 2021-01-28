import math

import matplotlib.pyplot as plt
import numpy as np


class OLD:
    @staticmethod
    def normpdf(x, mean, sd=1):
        var = float(sd) ** 2
        pi = 3.1415926
        denom = (2 * pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom

    @staticmethod
    def similarity(onset_prob, score_onset):
        sim = float(min(onset_prob, score_onset) + 1e-6) / (max(onset_prob, score_onset) + 1e-6)
        return sim

    @staticmethod
    def compute_f_I_given_D(fsource, f_I_J_given_D, cur_pos, scoreLen, window_size=200):  # 200
        left = max(0, cur_pos - window_size)
        right = min(scoreLen, cur_pos + window_size)
        f_I_given_D = np.zeros(scoreLen)
        fsource_w = fsource[left:right]
        f_I_J_given_D_w = f_I_J_given_D[:right - left]
        f_I_given_D_w = np.convolve(fsource_w, f_I_J_given_D_w)
        f_I_given_D_w = f_I_given_D_w / sum(f_I_given_D_w)
        if left + len(f_I_given_D_w) > scoreLen:
            end = scoreLen
        else:
            end = left + len(f_I_given_D_w)
        f_I_given_D[left:end] = f_I_given_D_w[:(end - left)]
        # plt.plot(f_I_given_D[:50])
        # plt.show()
        # plt.plot(fsource[:50])
        # plt.show()
        return f_I_given_D

    @staticmethod
    def compute_f_I_J_given_D(score_axis, estimated_tempo, elapsed_time, beta, alpha, Rc, no_move_flag):
        # if no_move_flag:
        #     print('no move')
        if estimated_tempo > 0:
            rateRatio = float(Rc) / float(estimated_tempo)
        else:
            rateRatio = Rc / 0.00001
        rateRatio = 1 / rateRatio
        sigmaSquare = math.log(float(1) / float(alpha * elapsed_time) + 1)
        sigma = math.sqrt(sigmaSquare)
        tmp1 = 1 / (score_axis * sigma * math.sqrt(2 * math.pi))
        tmp2 = (np.log(score_axis) - math.log(rateRatio * elapsed_time) + beta * sigmaSquare)
        tmp2 = np.exp(-tmp2 * tmp2 / (2 * sigmaSquare))
        distribution = tmp1 * tmp2
        distribution[score_axis <= 0] = 0

        distribution = distribution / sum(distribution)
        # for debug
        # plt.plot(distribution[:15])
        # plt.show()
        return distribution

    @staticmethod
    def compute_f_V_given_I(pitch, pitches, scoreLen, score_midi, onset_prob, score_onsets, alpha, w1, w2, w3, cur_pos,
                            std=1, WINSIZE=1, WEIGHT=[0.5]):
        # weight = 0.5 original
        # method2: diff with previous 5 pitches weighted as 0.1
        # WINSIZE = 5
        # WEIGHT = [0.1,0.1,0.1,0.1,0.1]
        reverse_judge = False
        f_V_given_I = np.zeros(scoreLen)
        sims = np.zeros(scoreLen)
        if pitch == -1:
            pitch = -1
        elif len(pitches) > WINSIZE:
            if pitch > 11.5:
                pitch_reverse = pitch - 12
                pitch_reverse = pitch_reverse - np.dot(pitches[-1 - WINSIZE:-1], WEIGHT)
                reverse_judge = True
            elif 0 < pitch < 0.5:
                pitch_reverse = pitch + 12
                pitch_reverse = pitch_reverse - np.dot(pitches[-1 - WINSIZE:-1], WEIGHT)
                reverse_judge = True
            pitch = pitch - np.dot(pitches[-1 - WINSIZE:-1], WEIGHT)

        # to check for two tempo at most per pitch
        # each i represent 0.01s
        window_size = 200  # 200
        left = max(0, cur_pos - window_size)
        right = min(scoreLen, cur_pos + window_size)

        for i in range(left, right):
            if score_midi[i] == -1:
                score_pitch = score_midi[i]
            # assert(score_midi[i] == -1,'fail with -1-1')
            # assert('fail with -1')
            # print('------------------------------------------1-1-1-1-1-1-')
            elif i >= WINSIZE:
                score_pitch = score_midi[i] - np.dot(score_midi[i - WINSIZE:i], WEIGHT)
            else:
                score_pitch = score_midi[i]
            score_onset = score_onsets[i]
            if pitch == -1:
                if score_pitch == -1:
                    f_V_given_I[i] = 0.1
                else:
                    f_V_given_I[i] = 0.00000000001
            elif score_pitch == -1:
                f_V_given_I[i] = 0.00000000001
            else:
                if reverse_judge and abs(pitch - score_pitch) > abs(pitch_reverse - score_pitch):
                    pitch = pitch_reverse
                f_V_given_I[i] = math.pow(
                    math.pow(OLD.normpdf(pitch, score_pitch, std), w1) * math.pow(
                        OLD.similarity(onset_prob, score_onset), w2), w3)

        return f_V_given_I


class NEW:
    @staticmethod
    def normalize(array):
        return np.true_divide(array, np.sum(array))

    @staticmethod
    def norm_pdf(x, mean, sd=1):
        var = sd ** 2
        denom = (2 * math.pi * var) ** 0.5
        num = math.exp(-(x - mean) ** 2 / (2 * var))
        return num / denom

    @staticmethod
    def similarity(left, right):
        return (min(left, right) + 1e-6) / (max(left, right) + 1e-6)

    @staticmethod
    def gate_mask(array, center, half_size):
        left_index = max(0, center - half_size)
        right_index = min(center + half_size, len(array))
        array[:left_index] = 0
        array[right_index:] = 0
        return array

    @staticmethod
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
        f_i_j_given_d = NEW.normalize(f_i_j_given_d)
        return f_i_j_given_d

    @staticmethod
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
        f_i_given_d = NEW.normalize(f_i_given_d)
        return f_i_given_d

    @staticmethod
    def compute_f_v_given_i(pitch_axis, onset_axis, cur_pos, axis_length, audio_pitch, audio_onset, pitch_proc, w):
        f_v_given_i = np.zeros(axis_length)
        left = max(0, cur_pos - 200)
        right = min(cur_pos + 200, axis_length)
        for i in range(left, right):
            if audio_pitch == -1:
                # performance side makes no sound
                if pitch_axis[i] == -1:
                    # score side also makes no sound
                    f_v_given_i[i] = 0.1
                else:
                    # score side makes sound
                    f_v_given_i[i] = 1e-11
            else:
                # performance side makes sound
                if pitch_axis[i] == -1:
                    # score side makes no sound
                    f_v_given_i[i] = 1e-11
                else:
                    # score side also makes sound
                    f_v_given_i[i] = math.pow(
                        math.pow(NEW.norm_pdf(pitch_proc.result(pitch_axis[i]), pitch_axis[i], 1), w[0])
                        * math.pow(NEW.similarity(audio_onset, onset_axis[i]), w[1]),
                        w[2]
                    )
        f_v_given_i = NEW.normalize(f_v_given_i)
        return f_v_given_i


if __name__ == '__main__':
    """environment check"""
    print('testing true division...', '1 / 3 =', 1 / 3)

    """useful constants"""
    param_original = {'marker': 'o', 'alpha': 0.75, 'color': 'blue', 'label': 'Original'}
    param_rewritten = {'marker': 'o', 'alpha': 0.75, 'color': 'green', 'label': 'Rewritten'}
    param_offset = {'marker': 'o', 'alpha': 0.75, 'color': 'red', 'label': 'Offset'}
    alpha = 10
    beta = 0.5
    offset_range = 1E-8
    resolution = 0.01
    chunk = 1024
    sr = 44100
    chunk_dur = chunk / sr

    """test similarity"""
    print(OLD.similarity(1, 0.5))
    print(NEW.similarity(1, 0.5))

    """test norm pdf"""
    x = np.linspace(90, 110, 100)
    y1 = [OLD.normpdf(x=x, mean=100, sd=1) for x in x]
    y2 = [NEW.norm_pdf(x=x, mean=100, sd=1) for x in x]
    plt.subplot(311)
    plt.title('norm pdf')
    plt.plot(x, np.array(y1), **param_original)
    plt.ylim([0, 1])
    plt.legend()
    plt.subplot(312)
    plt.plot(x, np.array(y2), **param_rewritten)
    plt.ylim([0, 1])
    plt.legend()
    plt.subplot(313)
    plt.plot(x, np.array(y1) - np.array(y2), **param_offset)
    plt.ylim([-offset_range, offset_range])
    plt.legend()
    plt.tight_layout()
    plt.show()

    """test f i j given d"""
    time_axis = np.arange(0, 1000, chunk_dur)
    score_tempo = 69
    estimated_tempo = 75
    elapsed_time = chunk_dur * 2
    old_val_1 = OLD.compute_f_I_J_given_D(score_axis=time_axis,
                                      estimated_tempo=estimated_tempo,
                                      elapsed_time=elapsed_time,
                                      beta=beta,
                                      alpha=alpha,
                                      Rc=score_tempo,
                                      no_move_flag=False)
    new_val_1 = NEW.compute_f_i_j_given_d(time_axis=time_axis,
                                          d=elapsed_time,
                                          score_tempo=score_tempo,
                                          estimated_tempo=estimated_tempo)
    trunc_1 = chunk_dur * 6
    plt.subplot(311)
    plt.title('f i j given d')
    plt.plot(time_axis, np.array(old_val_1), **param_original)
    plt.vlines(elapsed_time, 0, 1, color='red', label='Ref Pk')
    plt.vlines(time_axis[np.argmax(old_val_1)], 0, 1, color='blue', label='Act Pk')
    plt.xlim([0, trunc_1])
    plt.ylim([0, 1])
    plt.legend()
    plt.subplot(312)
    plt.plot(time_axis, np.array(new_val_1), **param_rewritten)
    plt.vlines(elapsed_time, 0, 1, color='red', label='Ref Pk')
    plt.vlines(time_axis[np.argmax(new_val_1)], 0, 1, color='blue', label='Act Pk')
    plt.xlim([0, trunc_1])
    plt.ylim([0, 1])
    plt.legend()
    plt.subplot(313)
    plt.plot(time_axis, np.array(old_val_1) - np.array(new_val_1), **param_offset)
    plt.xlim([0, trunc_1])
    plt.ylim([-offset_range, offset_range])
    plt.legend()
    plt.tight_layout()
    plt.show()

    """test f i given d"""
