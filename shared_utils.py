import os


def check_dir(*paths):
    for i in range(len(paths)):
        c_path = os.path.join(*paths[:i + 1])
        if not os.path.exists(c_path):
            os.mkdir(c_path)
        elif not os.path.isdir(c_path):
            return False
    return True


def average(array):
    return sum(array) / len(array)
