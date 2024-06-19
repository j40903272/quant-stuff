class Error(Exception):
    pass

class ParamError(Error):
    def __init__(self, message):
        self.message = message