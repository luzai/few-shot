from multiprocessing import Pool


def run(cls_instance, i):
    return cls_instance.func(i)


class Runner(object):
    def __init__(self):
        pool = Pool(processes=5)
        for i in range(5):
            pool.apply_async(run, (self, i))
        pool.close()
        pool.join()

    def func(self, i):
        print i
        return i


runner = Runner()