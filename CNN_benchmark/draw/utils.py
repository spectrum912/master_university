import os


def __result_dir_create(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)