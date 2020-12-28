import logging
from inspect import stack

class LoggerCls:

    # Always instantiate a logger, do not use the root logger, the top in hierarchy.
    # Beware do use logging_utils.basicConfig, initializes the root logger and overwrites every logger
    def __init__(self, log_type, name, filename, filemode, formatter, level):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.set_logger_type(log_type, filename, filemode, formatter)   # set type logger for each class instance

    def set_logger_type(self, log_type, filename, filemode, formatter):
        if log_type == "log_to_file":
            file_handler = logging.FileHandler(filename, filemode)
            formatter_ = logging.Formatter(formatter)
            file_handler.setFormatter(formatter_)
            self.logger.addHandler(file_handler)
        elif log_type == "log_to_stdout":
            stream_handler = logging.StreamHandler()
            formatter_ = logging.Formatter(formatter)
            stream_handler.setFormatter(formatter_)
            self.logger.addHandler(stream_handler)

    def _extended_format(self):
        _extended_format = " - Caller method: < " + stack()[2].function + ">" + \
                           " - Line No.: " + str(stack()[2].lineno)
        return _extended_format

    def debug(self, msg, **kwargs):
        self.logger.debug(msg + self._extended_format(), **kwargs)

    def info(self, msg, **kwargs):
        self.logger.info(msg + self._extended_format(), **kwargs)

    def warning(self, msg, **kwargs):
        self.logger.warning(msg + self._extended_format(), **kwargs)

    def exception(self, msg, **kwargs):
        self.logger.exception(msg, **kwargs)

    def error(self, msg, **kwargs):
        def exception(self, msg, **kwargs):
            self.logger.exception(msg, **kwargs)
        exception(self, msg,**kwargs)
