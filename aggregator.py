import numpy as np


class Updater:
    def __init__(self, kernel_stride, update_steps, sample_rate):
        self._update_size = int(sample_rate * kernel_stride)
        self._x = np.zeros((self._update_size * (update_steps - 1),))
        self._update = np.zeros((self._update_size,))
        self._n = 0
        self._update_steps = update_steps
        self._weights = np.ones((update_steps * self._update_size))

    def __call__(self, x):
        self._x = np.concatenate([self._x, self._update])
        self._x += (x - self._x) / self._weights
        y, self._x = np.split(self._x, [self._update_size])

        if self._n < self._update_steps:
            self._n += 1
            self._weights[:-self._n * self._update_size] += 1
        return y
