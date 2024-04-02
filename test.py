import datetime
import time


class Test:
    def __init__(self):
        self.Start = datetime.datetime.now()

class Test2(Test):
    def __init__(self):
        super().__init__()
        time.sleep(1)
        self.End = datetime.datetime.now()

    @property
    def ET(self):
        return (self.End - self.Start).total_seconds()

self = Test2()