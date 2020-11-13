import math

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as aa
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot


def norm_pdf(x, mean, sd=1):
    var = sd ** 2
    denom = (2 * math.pi * var) ** 0.5
    num = math.exp(-(x - mean) ** 2 / (2 * var))
    return num / denom


def comparer_0(input_now, input_prev, ref_now, ref_prev):
    if input_now == ref_now and input_prev == ref_prev:
        return 1
    else:
        return 0


def comparer_1(input_now, input_prev, ref_now, ref_prev):
    if 12 - input_now < 0.5:
        if abs(input_now - 12 - ref_now) < abs(input_now - ref_now):
            input_now -= 12
    elif input_now - 0 < 0.5:
        if abs(input_now + 12 - ref_now) < abs(input_now - ref_now):
            input_now += 12
    return norm_pdf(input_now - 0.5 * input_prev, ref_now - 0.5 * ref_prev)


if __name__ == '__main__':
    cur_ax = []
    prev_ax = []
    for i in range(12):
        for j in np.linspace(0, 12, 12, endpoint=False):
            cur_ax.append((j, i))
            prev_ax.append((j, i))
    result = np.zeros((len(prev_ax), len(cur_ax)))
    for i in range(len(prev_ax)):
        for j in range(len(cur_ax)):
            result[i, j] = comparer_0(cur_ax[j][0], prev_ax[i][0],
                                      cur_ax[j][1], prev_ax[i][1])

    host = host_subplot(111, axes_class=aa.Axes)
    par = host.twiny()
    new_fixed_axis = host.get_grid_helper().new_fixed_axis
    par.axis["top"] = new_fixed_axis(loc="top",
                                     axes=par,
                                     offset=(0, 10))
    par.set_xlabel("Ka")
    par.set_xlim((0, 15.71))
    par.axis["bottom"].set_visible(False)

    par = host.twinx()
    par.axis["right"] = new_fixed_axis(loc="right",
                                       axes=par,
                                       offset=(10, 0))
    par.axis["left"] = new_fixed_axis(loc="left",
                                      axes=par,
                                      offset=(-10, 0))
    host.imshow(result, cmap='Wistia')
    host.set_xlabel('Current')
    host.set_ylabel('Previous')
    plt.draw()
    plt.show()
