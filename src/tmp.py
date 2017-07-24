from pathos.multiprocessing import ProcessingPool


class Bar:
    def foo(self, name):
        return len(str(name))

    def boo(self, things):
        for thing in things:
            self.sum += self.foo(thing)
        return self.sum

    sum = 0


b = Bar()
results = ProcessingPool().map(b.boo, [[12, 3, 456], [8, 9, 10], ['a', 'b', 'cde']])
print  results
print b.sum
