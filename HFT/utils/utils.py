import os
import logging

class ColorFormatter(logging.Formatter):
    '''
    Color formatter wrapper
    '''

    def __init__(self, **kwargs):
        self._escape_char = '\033'
        self._reset = '{0}[0m'.format(self._escape_char)
        self._color_code = {
            'red': 31,
            'green': 32,
            'yellow': 33,
            'blue': 34,
            'magenta': 35,
            'cyan': 36,
            'white': 37
        }
        self._log_colors = {}
        self._fmt_str = ''
        self._date_fmt = ''
        for k, value in kwargs.items():
            if k in 'log_colors':
                self._log_colors = value
            elif k in 'format':
                self._fmt_str = value
            elif k in 'datefmt':
                self._date_fmt = value
            else:
                raise TypeError('unexpected keyword argument ' + k)

        logging.Formatter.__init__(self, self._fmt_str, datefmt=self._date_fmt)

    def format(self, record):
        levelname = record.levelname
        if levelname in self._log_colors.keys():
            temp_levelname = '{0}[{1}m{2}{3}'.format(
                self._escape_char,
                self._color_code[self._log_colors[levelname]],
                levelname.lower(), self._reset)
        record.levelname = temp_levelname
        return logging.Formatter.format(self, record)
    
def setup_logger(name, *, is_verbose=False, is_file_handled=False, file_dir=None):
    '''
    setup a logger for name
    '''
    log_name = name

    stream_handler = logging.StreamHandler()
    color_formatter = ColorFormatter(
        format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(funcName)s()] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'magenta'})
    stream_handler.setFormatter(color_formatter)

    logger = logging.getLogger(log_name)
    logger.addHandler(stream_handler)
    logger.setLevel('DEBUG' if is_verbose else 'INFO')
    logger.propagate = False

    if is_file_handled:
        normal_formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        file_folder = 'logger' if file_dir is None else file_dir
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        file_handler = logging.FileHandler(os.path.join(
            file_folder, '{0}.log'.format(log_name)), mode='a', encoding='utf-8')
        file_handler.setFormatter(normal_formatter)
        logger.addHandler(file_handler)
    return logger