import csv
from datetime import datetime


class CsvDealer:

    def __init__(self, file_path, logger, strategy_name, date_format='%Y-%m-%d', open_file=True):
        self._file_path = file_path
        self.logger = logger
        self._strategy_name = strategy_name
        self._date_format = date_format
        self._file = open(self._file_path, 'a', newline='') if open_file else None
        self._writer = csv.writer(self._file) if open_file else None

    @property
    def file_path(self):
        return self._file_path

    def open_file(self, new_file_path=None):
        if new_file_path is not None:
            self._file_path = new_file_path
        self._file = open(self._file_path, 'a', newline='')
        self._writer = csv.writer(self._file)

    def get_last_row_date_csv(self, format='%Y%m%d'):
        try:
            with open(self._file_path, "r", encoding="utf-8", errors="ignore") as scraped:
                last_row = None
                reader = csv.reader(scraped, delimiter=',')
                for row in reader:
                    if row:  # avoid blank lines
                        last_row = row
                return datetime.strptime(last_row[0], format) if last_row is not None else None
        except FileNotFoundError:
            None

    def write_to_file(self, data):
        if self._file is None:
            self.logger.info(f'[{self._strategy_name}] Writing csv file')
            with open(self._file_path, 'a', newline='') as csvfile:
                # add the csv writer
                if isinstance(data, str):
                    csvfile.write(data)
                else:
                    writer = csv.writer(csvfile)
                    writer.writerow(data)
        else:
            if isinstance(data, str):
                self._file.write(data)
            else:
                self._writer.writerow(data)

    def check_open_file(self, function):
        def wrapper():
            func = function()
            splitted_string = func.split()
            return splitted_string

        return wrapper

    def write_if_needed(self, msg, last_update_date):
        if isinstance(last_update_date, str):
            try:
                last_update_date = datetime.strptime(last_update_date, self._date_format)
            except ValueError:
                self.logger.error(f'Cannot transfer to dt - {last_update_date}')
                # no date is found, so write the data anyway
                last_update_date = None
        if isinstance(msg, str):
            data_dt = datetime.strptime(msg.split(',')[0], self._date_format) if last_update_date is not None else None
            if data_dt is None or last_update_date is None or last_update_date < data_dt:
                self._file.write(msg)
        else:
            # list
            data_dt = msg[0]
            if isinstance(data_dt, str):
                data_dt = datetime.strptime(msg[0], self._date_format) if last_update_date is not None else None
            if data_dt is None or last_update_date is None or last_update_date < data_dt:
                self._writer.writerow(msg)

    def close(self):
        self._file.close()

    def __del__(self):
        self.close()
