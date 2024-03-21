from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os

def closest_key(dct, comp):
    dist = 1e9
    for key, val in dct.items():
        temp = sum([(x-y)**2 for x, y in zip(comp, val)])
        if temp < dist:
            dist = temp
            out = key
    return out

def timeshape(dt):
    if dt.hour in [0, 1, 2, 3, 4, 5, 22, 23]:
        return '7x8'
    if dt.weekday() in [5, 6]:
        return '2x16'
    return '5x16'

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)