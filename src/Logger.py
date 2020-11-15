DEBUG = 0
INFO = 1
WARNING = 2
ERROR = 3
__Logger_dict = {
    DEBUG: 'DBG',
    INFO: 'INF',
    WARNING: 'WRN',
    ERROR: 'ERR'
}

def log(level=DEBUG, message=''):
    assert level >= 0 and level <= 3
    print('[{}] {}'.format(__Logger_dict[level], message))