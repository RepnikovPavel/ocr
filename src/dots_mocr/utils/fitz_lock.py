"""MuPDF operations must not run concurrently, even on separate documents.

Only rendering/preprocessing/overlays hold this lock; inference stays parallel.
An RLock permits image preprocessing to call the page renderer under the lock.
"""

from functools import wraps
from threading import RLock

_LOCK = RLock()


def serialized_fitz(function):
    @wraps(function)
    def locked(*args, **kwargs):
        with _LOCK:
            return function(*args, **kwargs)
    return locked
