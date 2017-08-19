import logging, os
from logging import INFO, DEBUG, WARN, ERROR
import utils


# todo log class for config to log
# log level: CRITICAL > ERROR > WARNING > INFO > DEBUG
class Filter(object):
  def __init__(self, level):
    self.level = level
  
  def filter(self, logRecord):
    return logRecord.levelno <= self.level


logger = logging.getLogger('global_logger')
# logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
# info record only info at info.log
info_handler = logging.FileHandler('info.log')
info_handler.setLevel(logging.INFO)
info_handler.addFilter(Filter(logging.INFO))  # just show INFO logs
info_handler.setFormatter(formatter)
logger.addHandler(info_handler)

# stream_handler show all logs into stdout screen, thus maybe easy for debug
stream_handler = logging.StreamHandler()
if utils.get_config('logger_level') == 'info':
  stream_handler.setLevel(logging.INFO)
elif utils.get_config('logger_level') == 'debug':
  stream_handler.setLevel(logging.DEBUG)
elif utils.get_config('logger_level') == 'error':
  stream_handler.setLevel(logging.ERROR)
elif utils.get_config('logger_level') == 'warning':
  stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# error_handler only record WARNING and ERROR and CRITICAL
error_handler = logging.FileHandler('err.log')
error_handler.setLevel(logging.WARNING)
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)

if __name__ == '__main__':
  print 'ok'