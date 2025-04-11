import sys
import time
from typing import Iterable, Any


class ProgressBar:
    """
    usage:

    # 1. fixed length object (when is has {__len__} method)
    for i in ProgressBar(list(range(10))):
        time.sleep(0.1)

    # 2. when you know total iteration count but object have no {__len__} method
    for i in ProgressBar(range(10), max_value=10):
        time.sleep(0.1)

    # 3. prefix, subfix usage
    task = ProgressBar(list(range(10)))
    for i in task:
        task.update(prefix='prefix', suffix='suffix')
        time.sleep(0.1)
    """

    def __init__(self, iterable: Iterable[Any], max_value: int = 0, bar_length: int = 30, prefix: str = '',
                 subfix: str = '', graph: bool = True):
        self.iterable = iterable
        self.max = max_value
        if (self.max == 0) and (hasattr(self.iterable, '__len__')):
            method = getattr(self.iterable, '__len__')
            self.max = method()
        self._prefix = prefix
        self._suffix = subfix
        self._i = 0
        self._bar_graph = '█'
        self._bar_length = bar_length
        self._iter = None
        self._start = time.time()
        self._graph = graph

    def __iter__(self):
        self._start = time.time()
        self._iter = iter(self.iterable)
        return self

    def update(self, prefix: str = None, suffix: str = None) -> None:
        self._prefix = self._prefix if prefix is None else prefix
        self._suffix = self._suffix if suffix is None else suffix

    @staticmethod
    def minutes_sec(sec) -> str:
        return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

    def _bar(self):
        sec = time.time() - self._start
        eta = (self.max - self._i) * (sec / max(self._i, 1))
        if self.max:
            dot_num = int(self._i / self.max * self._bar_length)
            dot = self._bar_graph * dot_num
            empty = '.' * (self._bar_length - dot_num)
            sys.stdout.write(
                f'\r {self._prefix} [time:{self.minutes_sec(sec)}] '
                f'[{dot}{empty}] {self._i}/{self.max}({self._i / self.max * 100:3.2f}%) '
                f'[ETA:{self.minutes_sec(eta)}] {self._suffix}')
        else:
            sys.stdout.write(f'\r[time:{self.minutes_sec(sec)}] {self._prefix} [ Status Unknown ... ] '
                             f'[{self._i}/??] {self._suffix}\n')

    def _text(self):
        sec = time.time() - self._start
        if self.max:
            if self._i % max((self.max // 5), 1) == 0:
                print(f'{self._prefix} [time:{self.minutes_sec(sec)}] '
                            f'{self._i:3d}/{self.max}({self._i / self.max * 100: 3.2f}%) {self._suffix}')

    def __next__(self) -> Any:
        if self._graph:
            self._bar()
        self._i += 1
        try:
            return next(self._iter)
        except StopIteration:
            raise StopIteration
