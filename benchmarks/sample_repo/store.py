"""A tiny in-memory key-value store with a dict behind it.

Part of the pinned benchmark workload (see ``benchmarks/README.md``).
"""


class Store:
    """A string-keyed store with a fixed capacity."""

    def __init__(self, capacity=100):
        self._data = {}
        self.capacity = capacity

    def put(self, key, value):
        """Store ``value`` under ``key``; evict the oldest key when full."""
        if len(self._data) >= self.capacity and key not in self._data:
            oldest = next(iter(self._data))
            del self._data[oldest]
        self._data[key] = value

    def get(self, key, default=None):
        """Return the value for ``key``, or ``default`` when absent."""
        return self._data.get(key, default)

    def delete(self, key):
        """Remove ``key``; return True when something was removed."""
        return self._data.pop(key, None) is not None

    def keys(self):
        """Return the stored keys in insertion order."""
        return list(self._data.keys())

    def __len__(self):
        return len(self._data)
