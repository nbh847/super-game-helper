import time
import os


def shutdown(seconds):
    print('{}s zhihouguanji'.format(seconds))
    time.sleep(seconds)
    os.system('shutdown -s -f -t 1')


shutdown(3600 * 9)
