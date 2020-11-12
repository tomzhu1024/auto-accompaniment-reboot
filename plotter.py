import matplotlib.pyplot as plt
import numpy as np


def plot_progress(dir_name):
    sf_progress_time = np.load('output/%s/sf_progress_time.npy' % dir_name)
    sf_progress_pos = np.load('output/%s/sf_progress_pos.npy' % dir_name)
    sf_progress_conf = np.load('output/%s/sf_progress_conf.npy' % dir_name)
    sf_progress_report = np.load('output/%s/sf_progress_report.npy' % dir_name)
    ac_progress_time = np.load('output/%s/ac_progress_time.npy' % dir_name)
    ac_progress_pos = np.load('output/%s/ac_progress_pos.npy' % dir_name)
    time_baseline = np.min(np.concatenate((sf_progress_time, ac_progress_time)))
    sf_progress_time -= time_baseline
    ac_progress_time -= time_baseline

    plt.scatter(sf_progress_time[sf_progress_report == 0], sf_progress_pos[sf_progress_report == 0], marker='.',
                c=(0.5 + sf_progress_conf)[sf_progress_report == 0], vmin=0, vmax=1, cmap='Blues', label='SF')
    plt.scatter(sf_progress_time[sf_progress_report == 1], sf_progress_pos[sf_progress_report == 1], marker='*',
                c=(0.5 + sf_progress_conf)[sf_progress_report == 1], vmin=0, vmax=1, cmap='Purples',
                label='SF (reported)')
    plt.scatter(ac_progress_time, ac_progress_pos, marker='.', color='orange', label='AC')

    plt.title('Progress Graph')
    plt.xlabel('Audio time (s)')
    plt.ylabel('Score time (s)')
    plt.legend()
    plt.show()


def plot_sf_lagging(dir_name):
    sf_time_cost = np.load('output/%s/sf_time_cost.npy' % dir_name)
    plt.plot(sf_time_cost * 1000, '.-', color='blue', alpha=0.25)
    plt.title('SF Lagging')
    plt.ylabel('Lagging (ms)')
    plt.show()


def plot_ac_lagging(dir_name):
    ac_progress_time = np.load('output/%s/ac_progress_time.npy' % dir_name)
    lagging = np.array([(ac_progress_time[i + 1] - ac_progress_time[i]) for i in range(len(ac_progress_time) - 1)])
    plt.plot(lagging * 1000, '.-', color='blue', alpha=0.25)
    plt.title('AC Lagging')
    plt.ylabel('Lagging (ms)')
    plt.show()


if __name__ == '__main__':
    my_config = {
        'name': '20201112-006'
    }
    plot_progress(my_config['name'])
    plot_sf_lagging(my_config['name'])
    plot_ac_lagging(my_config['name'])
