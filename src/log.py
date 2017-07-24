import logging, os
from opts import Config


# log level: CRITICAL > ERROR > WARNING > INFO > DEBUG
class Filter(object):
    def __init__(self, level):
        self.level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.level

if not os.path.exists(Config.output_path):
    os.mkdir(Config.output_path)
logger = logging.getLogger('global_logger')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s - %(message)s')

# info record only info at info.log
info_handler = logging.FileHandler(Config.output_path + '/info.log')
info_handler.setLevel(logging.INFO)
info_handler.addFilter(Filter(logging.INFO))  # just show INFO logs
info_handler.setFormatter(formatter)
logger.addHandler(info_handler)

# stream_handler show all logs into stdout screen, thus maybe easy for debug
if Config.stream_verbose:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

# error_handler only record WARNING and ERROR and CRITICAL
error_handler = logging.FileHandler(Config.output_path + '/err.log')
error_handler.setLevel(logging.WARNING)
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)
