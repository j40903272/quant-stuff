import logging
import datetime
import os


class Logger(object):

    def __init__(self, logger_folder="\crypto_trades\logs", file_name='', level=logging.INFO):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        if not os.path.exists(logger_folder):
            os.makedirs(logger_folder, exist_ok=True)

        if len(self.logger.handlers) == 0:
            formatter = logging.Formatter(
                fmt='[%(asctime)s %(levelname)-5s] %(message)s')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(logging.INFO)
            self.logger.addHandler(stream_handler)

            logger_file = os.path.join(
                logger_folder, "{}_{}.log".format(file_name, str(datetime.datetime.now().date())))
            file_handler = logging.FileHandler(logger_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def log(self, level, msg):
        self.logger.log(level, msg)

    def disable(self):
        logging.disable(50)
