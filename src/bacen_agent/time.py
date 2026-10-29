import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        initial = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} levou {time.perf_counter() - initial:.4f}s")
        return result
    return wrapper