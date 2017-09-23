import utils
from metadata import *
from utils import *

prefix = '/home/wangxinglu/prj/few-shot/data/imagenet-raw'
corrupt = read_list('/home/wangxinglu/prj/few-shot/src/corrupt')

os.chdir(prefix)
