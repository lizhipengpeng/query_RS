import time
import datetime

def get_time(start):
    return time.time() - start

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '
